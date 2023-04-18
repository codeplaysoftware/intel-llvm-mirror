//===--- InferActions.cpp - Address Space Inference check    ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "clang/InferASPChecker/InferActions.h"
#include "clang/InferASPChecker/ConstraintGather.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Mangle.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/InferASPChecker/ConstraintGather.h"
#include "clang/InferASPChecker/Unifier.h"
#include "clang/InferASPChecker/UnifyType.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <queue>
#include <utility>

using namespace clang;

namespace {

class CGWalker {
private:
  std::unique_ptr<CallGraph> CG;
  llvm::SmallPtrSet<const FunctionDecl *, 16> Visited;

public:
  CGWalker() = default;

  void reset(ASTContext &Context) { CG = std::make_unique<CallGraph>(); }

  void dump() { CG->dump(); }

  void addEntryPoint(FunctionDecl *DC) {
    assert(CG && "Graph not initialized");
    llvm::SmallVector<FunctionDecl *, 16> WorkList;
    CG->addToCallGraph(DC);

    auto FillWorkList = [&](CallGraphNode *CI) {
      if (!CI)
        return;
      llvm::for_each(CI->callees(), [&](const CallGraphNode::CallRecord &D) {
        if (auto *Callee =
                llvm::dyn_cast_or_null<FunctionDecl>(D.Callee->getDecl())) {
          Callee = Callee->getMostRecentDecl();
          if (!Visited.count(Callee))
            WorkList.push_back(Callee);
        }
      });
    };
    FillWorkList(CG->getNode(DC));

    while (!WorkList.empty()) {
      FunctionDecl *FD = WorkList.pop_back_val();
      if (!Visited.insert(FD).second)
        continue;
      CG->addToCallGraph(FD);
      FillWorkList(CG->getNode(FD));
    }
  }

  clang::CallGraph &get() {
    assert(CG && "Graph not initialized");
    return *CG.get();
  }
};

class ASPInferASTConsumerBase : public ASTConsumer {
  CGWalker Walker;
  llvm::SmallDenseSet<VarDecl *> GVDecl;
  ASTContext *Ctx;

  llvm::Timer InferenceTimer; // Inference Timer
  bool TimerEnabled;

  void startTimer() {
    if (TimerEnabled)
      InferenceTimer.startTimer();
  }

  void stopTimer() {
    if (TimerEnabled)
      InferenceTimer.stopTimer();
  }

public:
  ASPInferASTConsumerBase(CompilerInstance &CI)
      : Walker{}, Ctx{nullptr}, InferenceTimer{
                                    "syclinfer",
                                    "SYCL Address Space Inference Time"} {
    TimerEnabled = CI.getCodeGenOpts().TimePasses;
  }

protected:
  void Initialize(ASTContext &Context) override {
    Ctx = &Context;
    Walker.reset(Context);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    DiagnosticsEngine &Diags = Ctx->getDiagnostics();
    if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred())
      return true;

    // Collect debug info for all decls in this group.
    startTimer();
    for (auto *Decl : D) {
      if (auto *VD = dyn_cast<VarDecl>(Decl)) {
        GVDecl.insert(VD);
      }
      if (auto *FD = dyn_cast<FunctionDecl>(Decl)) {
        if (FD->hasAttr<SYCLDeviceAttr>() || FD->hasAttr<OpenCLKernelAttr>()) {
          Walker.addEntryPoint(FD);
        }
      }
    }
    stopTimer();

    return true;
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    DiagnosticsEngine &Diags = Ctx.getDiagnostics();
    if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred())
      return;

    startTimer();
    startASTWalk();

    llvm::for_each(GVDecl, [&](VarDecl *VD) { processVarDecl(VD); });

    for (auto &N : llvm::post_order(&Walker.get())) {
      Decl *D = N->getDecl();
      if (!D)
        continue;
      if (auto *FD = dyn_cast<FunctionDecl>(D)) {
        processDecl(FD);
      }
    }
    endASTWalk();
    stopTimer();
  }

  virtual void startASTWalk() {}
  virtual void processVarDecl(VarDecl *) {}
  virtual void processDecl(FunctionDecl *) {}
  virtual void endASTWalk() {}
};

class ASPInferConsumer : public ASPInferASTConsumerBase {
  unify::UnifyContext UnifyCtx;
  unify::FunctionLookup FuncLookup;
  unify::ConstraintGather Gather;

public:
  ASPInferConsumer(CompilerInstance &CI)
      : ASPInferASTConsumerBase(CI), UnifyCtx{CI.getASTContext(),
                                              CI.getDiagnostics()},
        Gather{UnifyCtx, CI.getASTContext(), FuncLookup} {}

protected:
  void startASTWalk() override {}

  void processVarDecl(VarDecl *VD) override { Gather.VisitVarDecl(VD); }

  void processDecl(FunctionDecl *FD) override { Gather.ProcessFunction(FD); }

  void endASTWalk() override {}
};

class GatherStatmentConsumer : public ASPInferASTConsumerBase {
  unify::StmtToHandle Gather;

public:
  GatherStatmentConsumer(CompilerInstance &CI) : ASPInferASTConsumerBase(CI) {}

protected:
  void processDecl(FunctionDecl *FD) override { Gather.ProcessFunction(FD); }
  void endASTWalk() override { Gather.print(llvm::outs()); }
};

} // end anonymous namespace

InferASPAction::InferASPAction(std::unique_ptr<FrontendAction> WrappedAction)
    : WrapperFrontendAction(std::move(WrappedAction)) {}

std::unique_ptr<ASTConsumer>
InferASPAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  // our consumer should be before the wrapped one.
  // If the wrapped one is the codegen, the AST may be deleted so we would have
  // dandling pointers.
  if (CI.getASTContext().getLangOpts().SYCLInferASPDumpStatementKind)
    Consumers.push_back(std::make_unique<GatherStatmentConsumer>(CI));
  Consumers.push_back(std::make_unique<ASPInferConsumer>(CI));
  Consumers.push_back(WrapperFrontendAction::CreateASTConsumer(CI, InFile));
  return std::make_unique<MultiplexConsumer>(std::move(Consumers));
}

bool InferASPAction::BeginInvocation(CompilerInstance &CI) {
  return WrapperFrontendAction::BeginInvocation(CI);
}

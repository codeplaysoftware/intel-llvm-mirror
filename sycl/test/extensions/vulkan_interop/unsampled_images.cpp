// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %link-vulkan %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include "vulkan_common.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

void run_sycl(int input_image_fd, int output_image_fd, size_t width,
              size_t height) {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Image descriptor - mapped to Vulkan image layout
  sycl::ext::oneapi::image_descriptor desc(
      {width, height}, sycl::image_channel_order::rgba,
      sycl::image_channel_type::unsigned_int32);

  const size_t imgSize = width * height * sizeof(sycl::uint4);

  // Extension: external memory descriptor
  sycl::ext::oneapi::external_mem_descriptor inputExtMemDesc;
  sycl::ext::oneapi::external_mem_descriptor outputExtMemDesc;

  inputExtMemDesc.handle.fd = input_image_fd;
  outputExtMemDesc.handle.fd = output_image_fd;

  // Extension: interop mem handle imported from external memory
  sycl::ext::oneapi::interop_mem_handle img_input_mem_handle =
      sycl::ext::oneapi::import_external_memory(
          ctxt, imgSize, inputExtMemDesc,
          sycl::ext::oneapi::external_memory_type::OpaqueFD);

  sycl::ext::oneapi::interop_mem_handle img_output_mem_handle =
      sycl::ext::oneapi::import_external_memory(
          ctxt, imgSize, outputExtMemDesc,
          sycl::ext::oneapi::external_memory_type::OpaqueFD);

  // Extension: create the image and return the handle
  sycl::ext::oneapi::unsampled_image_handle img_input =
      sycl::ext::oneapi::create_image_interop(ctxt, img_input_mem_handle, desc);
  sycl::ext::oneapi::unsampled_image_handle img_output =
      sycl::ext::oneapi::create_image_interop(ctxt, img_output_mem_handle,
                                              desc);

  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class image_interop>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Extension: read image data from handle (Vulkan imported)
            sycl::uint4 pixel = sycl::ext::oneapi::read_image<sycl::uint4>(
                img_input, sycl::int2(dim0, dim1));

            pixel *= 10;

            // Extension: write image data using handle (Vulkan imported)
            sycl::ext::oneapi::write_image(img_output, sycl::int2(dim0, dim1),
                                           pixel);
          });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  try {
    sycl::ext::oneapi::destroy_external_memory(ctxt, img_input_mem_handle);
    sycl::ext::oneapi::destroy_external_memory(ctxt, img_output_mem_handle);
    sycl::ext::oneapi::destroy_image_handle(ctxt, img_input);
    sycl::ext::oneapi::destroy_image_handle(ctxt, img_output);
  } catch (...) {
    std::cerr << "Destroying interop memory failed!\n";
  }
}

void run_test() {
  const uint32_t width = 8, height = 8;
  const size_t imageSizeBytes = width * height * sizeof(sycl::uint4);

  std::cout << "Creating input image\n";
  // Create input image memory
  auto inputImage = vkutil::createImage(
      VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, {width, height, 1},
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_STORAGE_BIT);
  auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      inputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  auto inputMemory =
      vkutil::allocateDeviceMemory(imageSizeBytes, inputImageMemoryTypeIndex);
  VK_CHECK_CALL(vkBindImageMemory(vk_device, inputImage, inputMemory,
                                  0 /*memoryOffset*/));

  std::cout << "Creating output image\n";
  // Create output image memory
  auto outputImage = vkutil::createImage(
      VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, {width, height, 1},
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_STORAGE_BIT);
  auto outputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      outputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  auto outputMemory =
      vkutil::allocateDeviceMemory(imageSizeBytes, outputImageMemoryTypeIndex);
  VK_CHECK_CALL(vkBindImageMemory(vk_device, outputImage, outputMemory,
                                  0 /*memoryOffset*/));

  std::cout << "Creating staging buffers\n";
  // Create input staging memory
  auto inputStagingBuffer = vkutil::createBuffer(
      imageSizeBytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto inputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
      inputStagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto inputStagingMemory = vkutil::allocateDeviceMemory(
      imageSizeBytes, inputStagingMemoryTypeIndex, false /*exportable*/);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, inputStagingBuffer,
                                   inputStagingMemory, 0 /*memoryOffset*/));

  // Create output staging memory
  auto outputStagingBuffer = vkutil::createBuffer(
      imageSizeBytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto outputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
      outputStagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto outputStagingMemory = vkutil::allocateDeviceMemory(
      imageSizeBytes, outputStagingMemoryTypeIndex, false /*exportable*/);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, outputStagingBuffer,
                                   outputStagingMemory, 0 /*memoryOffset*/));

  std::cout << "Populating staging buffer\n";
  // Populate staging memory
  sycl::vec<uint32_t, 4> *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inputStagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < width * height; ++i) {
    inputStagingData[i] =
        sycl::vec<uint32_t, 4>{4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3};
  }
  vkUnmapMemory(vk_device, inputStagingMemory);

  std::cout << "Submitting image layout transition\n";
  // Transition image layouts
  {
    VkImageMemoryBarrier barrierInput = {};
    barrierInput.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrierInput.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrierInput.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierInput.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierInput.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierInput.image = inputImage;
    barrierInput.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrierInput.subresourceRange.levelCount = 1;
    barrierInput.subresourceRange.layerCount = 1;
    barrierInput.srcAccessMask = 0;
    barrierInput.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkImageMemoryBarrier barrierOutput = {};
    barrierOutput.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrierOutput.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrierOutput.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierOutput.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierOutput.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierOutput.image = outputImage;
    barrierOutput.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrierOutput.subresourceRange.levelCount = 1;
    barrierOutput.subresourceRange.layerCount = 1;
    barrierOutput.srcAccessMask = 0;
    barrierOutput.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_computeCmdBuffer, &cbbi));
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput);
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierOutput);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_computeCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_computeCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_compute_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_compute_queue));
  }

  std::cout << "Copying staging memory to images\n";
  // Copy staging to main image memory
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffer, &cbbi));
    vkCmdCopyBufferToImage(vk_transferCmdBuffer, inputStagingBuffer, inputImage,
                           VK_IMAGE_LAYOUT_GENERAL, 1 /*regionCount*/,
                           &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
  }

  std::cout << "Getting memory file descriptors and calling into SYCL\n";
  // Pass memory to SYCL for modification
  auto input_fd = vkutil::getMemoryOpaqueFD(inputMemory);
  auto output_fd = vkutil::getMemoryOpaqueFD(outputMemory);
  run_sycl(input_fd, output_fd, width, height);

  std::cout << "Copying image memory to staging memory\n";
  // Copy main image memory to staging
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffer, &cbbi));
    vkCmdCopyImageToBuffer(vk_transferCmdBuffer, outputImage,
                           VK_IMAGE_LAYOUT_GENERAL, outputStagingBuffer,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
  }

  std::cout << "Validating\n";
  // Validate that SYCL made changes to the memory
  bool validated = true;
  sycl::vec<uint32_t, 4> *outputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, outputStagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&outputStagingData));
  for (int i = 0; i < width * height; ++i) {
    sycl::vec<uint32_t, 4> expected = {4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3};
    expected *= 10;
    for (int j = 0; j < 4; ++j) {
      if (outputStagingData[i][j] != expected[j]) {
        std::cerr << "Result mismatch! actual[" << i << "][" << j
                  << "] == " << outputStagingData[i][j]
                  << " : expected == " << expected[j] << "\n";
        validated = false;
        break;
      }
    }
    if (!validated)
      break;
  }
  vkUnmapMemory(vk_device, outputStagingMemory);

  if (validated) {
    std::cout << "Results are correct!\n";
  }

  // Cleanup
  vkDestroyBuffer(vk_device, inputStagingBuffer, nullptr);
  vkDestroyBuffer(vk_device, outputStagingBuffer, nullptr);
  vkDestroyImage(vk_device, inputImage, nullptr);
  vkDestroyImage(vk_device, outputImage, nullptr);
  vkFreeMemory(vk_device, inputStagingMemory, nullptr);
  vkFreeMemory(vk_device, outputStagingMemory, nullptr);
  vkFreeMemory(vk_device, inputMemory, nullptr);
  vkFreeMemory(vk_device, outputMemory, nullptr);
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupDevice() != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return EXIT_FAILURE;
  }

  run_test();

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

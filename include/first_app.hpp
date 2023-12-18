#pragma once

#include <memory>
#include <vector>

#include <lve_pipeline.hpp>
#include <lve_window.hpp>
#include <lve_device.hpp>
#include <lve_swap_chain.hpp>

namespace lve {
    class FirstApp {
       public:
        static constexpr int WIDTH = 1200;
        static constexpr int HEIGHT = 800;
        static constexpr std::string TITLE = "LearnVulkan";

        FirstApp();
        ~FirstApp();

        FirstApp(const FirstApp &) = delete;
        FirstApp &operator=(const FirstApp &) = delete;

        void run();

       private:
        void createPipelineLayout();
        void createPipeline();
        void createCommandBuffers();
        void drawFrame();

        LveWindow lveWindow{WIDTH, HEIGHT, TITLE};
        LveDevice lveDevice{lveWindow};
        LveSwapChain lveSwapChain{lveDevice, lveWindow.getExtent()};
        std::unique_ptr<LvePipeline> lvePipeline;
        VkPipelineLayout pipelineLayout;
        std::vector<VkCommandBuffer> commandBuffers;
    };

}  // namespace lve
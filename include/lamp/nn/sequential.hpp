/**
 * @file sequential.hpp
 * @brief Sequential container for neural network modules.
 * @author Gabriele Pintus
 * @version 1.0
 */

#pragma once

#include "lamp/nn/module.hpp"
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace lamp {
namespace nn {

/**
 * @class Sequential
 * @brief A sequential container that chains multiple modules together.
 *
 * Sequential applies a series of modules in order, where the output of
 * one module becomes the input to the next. This is useful for building
 * feed-forward neural networks.
 *
 * Example usage:
 * @code
 * auto model = Sequential({
 *     std::make_shared<Linear>(784, 128),
 *     std::make_shared<ReLU>(),
 *     std::make_shared<Linear>(128, 10)
 * });
 * @endcode
 */
class Sequential : public Module {
public:
    /**
     * @brief Default constructor creates an empty sequential container.
     */
    Sequential() = default;

    /**
     * @brief Construct a Sequential container with a list of modules.
     *
     * @param modules Initializer list of shared pointers to modules.
     */
    Sequential(std::initializer_list<std::shared_ptr<Module>> modules);

    /**
     * @brief Performs forward pass through all modules in sequence.
     *
     * Applies each module in order: output = module_n(...module_2(module_1(input)))
     *
     * @param input The input tensor.
     * @return Tensor The output after passing through all modules.
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Zeros gradients for all submodules.
     *
     * Calls zero_grad() on each contained module.
     */
    void zero_grad() override;

    /**
     * @brief Add a module to the sequential container.
     *
     * @param name The name identifier for the module.
     * @param module Shared pointer to the module to add.
     */
    void add_module(const std::string& name, const std::shared_ptr<Module>& module);

private:
    /**
     * @brief Registers all parameters from a submodule with this module.
     *
     * @param prefix Name prefix for the parameters.
     * @param module The submodule whose parameters to register.
     */
    void register_submodule_parameters(const std::string& prefix, Module& module);

    std::vector<std::shared_ptr<Module>> modules_;  ///< Ordered list of modules
};

} // namespace nn
} // namespace lamp
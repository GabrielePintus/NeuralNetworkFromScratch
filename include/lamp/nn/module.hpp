/**
 * @file module.hpp
 * @brief Base class for all neural network modules.
 * @author Gabriele Pintus
 * @version 1.0
 */

#pragma once

#include "lamp/core/tensor.hpp"
#include <unordered_map>
#include <string>

namespace lamp {
namespace nn {

/**
 * @class Module
 * @brief Abstract base class for all neural network modules.
 *
 * This class provides the foundation for building neural network layers and models.
 * It manages parameters, training/evaluation modes, and gradients.
 *
 * All neural network layers should inherit from this class and implement the
 * forward() method.
 */
class Module {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Module();

    /**
     * @brief Performs the forward pass computation.
     *
     * This pure virtual function must be implemented by derived classes
     * to define the layer's computation.
     *
     * @param input The input tensor.
     * @return Tensor The output tensor after computation.
     */
    virtual Tensor forward(const Tensor& input) = 0;

    /**
     * @brief Convenience operator for calling forward().
     *
     * Allows using the module as a function: output = module(input).
     *
     * @param input The input tensor.
     * @return Tensor The output tensor.
     */
    Tensor operator()(const Tensor& input);

    /**
     * @brief Sets the module to training mode.
     *
     * In training mode, layers like Dropout behave differently.
     */
    void train();

    /**
     * @brief Sets the module to evaluation mode.
     *
     * In evaluation mode, layers like Dropout are disabled.
     */
    void eval();

    /**
     * @brief Checks if the module is in training mode.
     *
     * @return bool True if in training mode, false otherwise.
     */
    bool is_training() const;

    /**
     * @brief Zeros out all parameter gradients.
     *
     * This method should be called before each backward pass to reset
     * accumulated gradients.
     */
    virtual void zero_grad();

    /**
     * @brief Registers a parameter tensor with the module.
     *
     * Registered parameters are included in gradient updates and
     * can be accessed via the parameters() method.
     *
     * @param name The parameter name.
     * @param param Reference to the parameter tensor.
     */
    void register_parameter(const std::string& name, Tensor& param);

    /**
     * @brief Access all registered parameters.
     *
     * @return std::unordered_map<std::string, Tensor*>& Map of parameter names to tensors.
     */
    std::unordered_map<std::string, Tensor*>& parameters();

protected:
    bool training_ = true;  ///< Training mode flag
    std::unordered_map<std::string, Tensor*> parameters_;  ///< Registered parameters
};

} // namespace nn
} // namespace lamp
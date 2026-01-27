#pragma once
#include "tensor.hpp"
#include <memory>
#include <unordered_map>
#include <string>

namespace lamp {

class Module {
public:
    virtual ~Module() = default;
    
    // Forward pass - pure virtual, must be implemented
    virtual Tensor forward(const Tensor& input) = 0;
    
    // Convenience operator for forward pass
    Tensor operator()(const Tensor& input) { return forward(input); }
    
    // Training mode management
    void train() { training_ = true; }
    void eval() { training_ = false; }
    bool is_training() const { return training_; }
    
    // Parameter management
    void register_parameter(const std::string& name, Tensor& param) {
        parameters_[name] = &param;
    }
    
    std::unordered_map<std::string, Tensor*>& parameters() {
        return parameters_;
    }

protected:
    bool training_ = true;
    std::unordered_map<std::string, Tensor*> parameters_;
};

} // namespace lamp
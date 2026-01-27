#pragma once


#include "tensor.hpp"
#include <unordered_map>
#include <string>


namespace lamp {


class Module {
public:
virtual ~Module();


// Forward pass - pure virtual, must be implemented
virtual Tensor forward(const Tensor& input) = 0;


// Convenience operator for forward pass
Tensor operator()(const Tensor& input);


// Training mode management
void train();
void eval();
bool is_training() const;


// Gradient management
virtual void zero_grad();


// Parameter management
void register_parameter(const std::string& name, Tensor& param);
std::unordered_map<std::string, Tensor*>& parameters();


protected:
bool training_ = true;
std::unordered_map<std::string, Tensor*> parameters_;
};


} // namespace lamp
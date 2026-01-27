#include "sequential.hpp"
#include <string> // Required for std::to_string

namespace lamp {

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    size_t index = 0;
    for (const auto& module : modules) {
        add_module(std::to_string(index++), module);
    }
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& module : modules_) {
        // Dereference the shared_ptr to call the operator() on the Module
        output = (*module)(output);
    }
    return output;
}

void Sequential::zero_grad() {
    // Call the base class implementation
    Module::zero_grad();
    
    // Zero out gradients for all submodules
    for (const auto& module : modules_) {
        module->zero_grad();
    }
}

void Sequential::add_module(const std::string& name, const std::shared_ptr<Module>& module) {
    modules_.push_back(module);
    register_submodule_parameters(name, *module);
}

void Sequential::register_submodule_parameters(const std::string& prefix, Module& module) {
    for (const auto& entry : module.parameters()) {
        register_parameter(prefix + "." + entry.first, *entry.second);
    }
}

} // namespace lamp
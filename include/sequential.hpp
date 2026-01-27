#pragma once

#include "module.hpp"
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lamp {

class Sequential : public Module {
public:
    Sequential() = default;

    Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
        size_t index = 0;
        for (const auto& module : modules) {
            add_module(std::to_string(index++), module);
        }
    }

    Tensor forward(const Tensor& input) override {
        Tensor output = input;
        for (const auto& module : modules_) {
            output = (*module)(output);
        }
        return output;
    }

    void zero_grad() override {
        Module::zero_grad();
        for (const auto& module : modules_) {
            module->zero_grad();
        }
    }

    void add_module(const std::string& name, const std::shared_ptr<Module>& module) {
        modules_.push_back(module);
        register_submodule_parameters(name, *module);
    }

private:
    void register_submodule_parameters(const std::string& prefix, Module& module) {
        for (const auto& entry : module.parameters()) {
            register_parameter(prefix + "." + entry.first, *entry.second);
        }
    }

    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace lamp
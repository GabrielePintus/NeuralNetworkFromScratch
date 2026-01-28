#pragma once

#include "lamp/nn/module.hpp"
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace lamp {
namespace nn {

class Sequential : public Module {
public:
    Sequential() = default;

    Sequential(std::initializer_list<std::shared_ptr<Module>> modules);

    Tensor forward(const Tensor& input) override;

    void zero_grad() override;

    void add_module(const std::string& name, const std::shared_ptr<Module>& module);

private:
    void register_submodule_parameters(const std::string& prefix, Module& module);

    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace nn
} // namespace lamp
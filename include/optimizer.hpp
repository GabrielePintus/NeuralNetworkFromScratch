#pragma once


#include "module.hpp"
#include <vector>


namespace lamp {


class SGD {
public:
SGD(Module& module, float learning_rate);
SGD(const std::vector<Tensor*>& params, float learning_rate);


void step();
void zero_grad();


private:
std::vector<Tensor*> params_;
float learning_rate_;
};


} // namespace lamp
#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_RandomNormalLike.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseRandomNormalLike = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type = ETensorType::UNDEFINED;
   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser RandomNormalLike op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);
   float attr_mean = 0;
   float attr_scale = 1.0;
   int attr_dtype = -1;
   float attr_seed = -1;

    for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
        std::string attribute_name = nodeproto.attribute(i).name();
        if (attribute_name == "mean")
            attr_mean = nodeproto.attribute(i).f();
        if (attribute_name == "scale")
            attr_scale = nodeproto.attribute(i).f();
        if (attribute_name == "dtype")
            attr_dtype = nodeproto.attribute(i).i();
        if (attribute_name == "seed")
            attr_seed = nodeproto.attribute(i).f();
    }
    if(attr_dtype != -1){
        if(attr_seed != -1)
            op.reset(new ROperator_RandomNormalLike<float>(attr_dtype, attr_mean, attr_scale, attr_seed, input_name, output_name));
        else 
            op.reset(new ROperator_RandomNormalLike<float>(attr_dtype, attr_mean, attr_scale, input_name, output_name));
    }
    else if(attr_seed != -1)
        op.reset(new ROperator_RandomNormalLike<float>(attr_mean, attr_scale, attr_seed, input_name, output_name));
    else 
        op.reset(new ROperator_RandomNormalLike<float>(attr_mean, attr_scale, input_name, output_name));

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

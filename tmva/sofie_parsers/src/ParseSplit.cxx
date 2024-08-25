#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/ROperator_Split.hxx"
#include "onnx_proto3.pb.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

ParserFuncSignature ParseSplit = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Split op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::string split_name;
   if (nodeproto.input_size() > 1) {
      split_name = nodeproto.input(1);
        parser.RegisterTensorType(split_name,input_type); // this should not be here ig
      if (!parser.IsRegisteredTensorType(split_name)) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Split op has input tensor" + split_name +
                                  " but its type is not yet registered");
      }
   }

   int attr_axis = 0;
   int attr_num_outputs = 0;
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis")
         attr_axis = nodeproto.attribute(i).i();
      else if (attribute_name == "num_outputs")
         attr_num_outputs = nodeproto.attribute(i).i();
   }

   // number of splits are given by the number of output tensors
   size_t output_size = nodeproto.output_size();
   std::vector<std::string> output_names(output_size);
   for (size_t i = 0; i < output_size; i++)
      output_names[i] = nodeproto.output(i);
   
   std::unique_ptr<ROperator> op;
   if (nodeproto.attribute_size()>1) {
    op.reset(new ROperator_Split<float>(attr_axis,attr_num_outputs, input_name, output_names));
   } else {
    op.reset(new ROperator_Split<float>(attr_axis,split_name, input_name, output_names));
   }

   for (size_t i = 0; i < output_size; i++) {
      if (!parser.IsRegisteredTensorType(output_names[i])) {
        parser.RegisterTensorType(output_names[i], input_type);
      }
   }

   return op;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
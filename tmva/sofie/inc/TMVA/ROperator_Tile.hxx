#ifndef TMVA_SOFIE_ROPERATOR_Tile
#define TMVA_SOFIE_ROPERATOR_Tile

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Tile final : public ROperator
{

private:

   std::string fNInput;
   std::string fNRepeats;
   std::string fNY;
   std::vector<Dim>fShapeInput;
   std::vector<Dim> fShapeY;
   bool fIsDynamic = false;

public:
   ROperator_Tile(){}
   ROperator_Tile(std::string nameRepeat, std::string nameInput, std::string nameY):
      fNRepeats(UTILITY::Clean_name(nameRepeat)),fNInput(UTILITY::Clean_name(nameInput)), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<Dim>> ShapeInference(std::vector<std::vector<Dim>> input){
      std::vector<Dim> ret = input[0];
      ret.isParam = true;

      for(size_t i=0; i < input[1].size(); i++) {
         if(!fIsDynamic)
            ret[i].dim=ret[i].dim*input[1][i].dim;
         else
            ret[i].param=ret[i].param + "*" + input[1][i].GetVal();
      }
      return {ret};
   }

   void Initialize(RModel& model){
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNInput) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      if(model.isInitializedTensor(fNInputs) || model.isDynamicTensor(fNInputs))
         fIsDynamic = true;
      fShapeInput=model.GetDynamicTensorShape(fNInput);
     // std::cout<<model.IsInitializedTensor(fNRepeats)<<"\n";

         // Retrieve the data pointer for the repeats tensor
      auto repptr = model.GetInitializedTensorData(fNRepeats);
      // Cast the raw pointer to the appropriate type (size_t*)
      auto repeat_shape = static_cast<size_t*>(repptr.get());

      if (repeat_shape == nullptr) {
        throw std::runtime_error("Failed to retrieve the data for the repeats tensor.");
      }
      // Get the shape of the repeats tensor to determine the number of elements
      auto repeats_shape = model.GetTensorShape(fNRepeats);
      // Ensure the repeats tensor is 1D and get the number of elements
      if (repeats_shape.size() != 1) {
         throw std::runtime_error("Repeats tensor is not 1D.");
      }
      size_t num_elements = repeats_shape[0];
      // Convert the data to a vector
      std::vector<size_t> repeats_vector(repeat_shape, repeat_shape + num_elements);

      // std::vector<size_t> repeat_shape=model.GetTensorShape(fNRepeats);
      fShapeY = ShapeInference({fShapeInput,ConvertShapetoDim(repeats_vector)})[0];
      //for(auto i : fShapeY) std::cout << i << " ";
     // std::cout << std::endl;
  
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty()) {
            throw std::runtime_error("TMVA SOFIE Tile Op called to Generate without being initialized first");
      }

      std::string input_length = ConvertDynamicShapeToLength(fShapeInput);
      std::string output_length = ConvertDynamicShapeToLength(fShapeY);

      std::stringstream out;
      out << "///-------- Tile operator\n";
      out << "std::vector<size_t> input_shape = " << ConvertDynamicShapeToString(fShapeInput) << ";\n";
      out << "std::vector<size_t> output_shape = " << ConvertDynamicShapeToString(fShapeY) << ";\n";
      out << "std::vector<size_t> indices(input_shape.size(), 0);\n";
      out << "for (size_t i = 0; i < " << output_length << "; ++i) {\n";
      out << SP<<"size_t source_index = 0;\n";
      out << SP<<"size_t stride = 1;\n";
      out << SP<<"for (int j = input_shape.size() - 1; j >= 0; --j) {\n";
      out << SP<<SP<<"source_index += (indices[j] % input_shape[j]) * stride;\n";
      out << SP<<SP<<"stride *= input_shape[j];\n";
      out << SP<<"}\n";
      out << SP<<"tensor_"<<fNY<<"[i] = tensor_"<<fNInput<<"[source_index];\n";
      out << SP<<"for (int j = input_shape.size() - 1; j >= 0; --j) {\n";
      out << SP<<SP<<"if (++indices[j] < output_shape[j]) {\n";
      out << SP<<SP<<SP<<"break;\n";
      out << SP<<SP<<"}\n";
      out << SP<<SP<<"indices[j] = 0;\n";
      out << SP<<"}\n";
      out << "}\n";

      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_Tile

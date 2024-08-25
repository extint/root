#ifndef TMVA_SOFIE_ROPERATOR_Split
#define TMVA_SOFIE_ROPERATOR_Split

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Split final : public ROperator
{

private:

   int fAttrAxis;
   int fAttrNum_outputs = 0;
   bool equal_tensors = false;

   std::string nameSplit;
   std::string fNX;
   std::vector<std::string> fNYs;
   std::vector<size_t> fShapeX;
   std::vector<std::vector<size_t>> fShapesY;
   std::vector<size_t> fNSplits;

public:
   ROperator_Split(){}
   ROperator_Split(int axis, int num_outputs, const std::string nameX, const std::vector<std::string> &  namesY):
        fAttrAxis(axis), fAttrNum_outputs(num_outputs), fNX(UTILITY::Clean_name(nameX)){
            fNYs.reserve(namesY.size());
            for (auto & name : namesY)
                fNYs.push_back(UTILITY::Clean_name(name));
            equal_tensors = true;
            std::cout<<"here1\n";
        }
   ROperator_Split(int axis, const std::string split, const std::string nameX, const std::vector<std::string> &  namesY):
        fAttrAxis(axis), nameSplit(UTILITY::Clean_name(split)), fNX(UTILITY::Clean_name(nameX)){
            fNYs.reserve(namesY.size());
            for (auto & name : namesY)
                fNYs.push_back(UTILITY::Clean_name(name));
            std::cout<<"here2\n";
            
        }
    

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
        std::vector<std::vector<size_t>> ret;
        if(equal_tensors){
            ret.resize(fAttrNum_outputs);
            float target_axis_value = input[0][fAttrAxis];
            const float axis_slice = ceil(target_axis_value/fAttrNum_outputs);
            for(size_t i = 0; i < ret.size(); i++){
                ret[i] = input[0];
                ret[i][fAttrAxis] = std::min(axis_slice, target_axis_value);
                target_axis_value -= axis_slice;
            }
        } else {
            ret.resize(fNSplits.size());
            for(size_t i = 0; i < ret.size(); i++){
                ret[i] = input[0];
                ret[i][fAttrAxis] = fNSplits[i];
            }
        }
        return ret;
   }

   void Initialize(RModel& model){
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("TMVA SOFIE Split Op Input Tensor is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (fAttrAxis < 0) fAttrAxis = fShapeX.size() + fAttrAxis;
      if (fAttrAxis < 0 || fAttrAxis >= (int) fShapeX.size())
          throw std::runtime_error("TMVA SOFIE Split Op - invalid axis value ");
      if (fAttrNum_outputs < 0 || fAttrNum_outputs > (int) fShapeX[fAttrAxis])
          throw std::runtime_error("TMVA SOFIE Split Op - invalid num_outputs value ");
          
      if(!equal_tensors){
        if (model.CheckIfTensorAlreadyExist(nameSplit) == false){
            throw std::runtime_error("TMVA SOFIE Split Op splits Tensor is not found in model");
        }

        const auto split_ptr = model.GetInitializedTensorData(nameSplit);
        const auto shape = model.GetTensorShape(nameSplit);
        const auto split_shape = static_cast<int64_t*>(split_ptr.get());
        fNSplits.assign(split_shape, split_shape + model.GetTensorShape(nameSplit)[0]);
        
        size_t sum = 0;
        for(auto i : fNSplits) sum += i;
        
        if(sum != fShapeX[fAttrAxis])
            throw std::runtime_error("TMVA SOFIE Split Op - invalid splits value ");    
        if (fNSplits.size() > fShapeX[fAttrAxis])
            throw std::runtime_error("TMVA SOFIE Split Op - invalid splits value ");
      }
        fShapesY = ShapeInference({fShapeX});
      for (size_t i = 0; i < fNYs.size(); i++) {
        std::cout << "Split - output shape " << ConvertShapeToString(fShapesY[i]) << std::endl;
        model.AddIntermediateTensor(fNYs[i], model.GetTensorType(fNX), fShapesY[i]);
      }
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA SOFIE Split operator called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ Split\n";
      // fshapeX = [3,2,10,4], fAttrAxis = 2, num_outputs = 3
      // fShapeY : [[3,2,4,4], [3,2,4,4], [3,2,2,4]]
      // out_offsets = [240,80,40,4]
      size_t offset = 1;
      for(size_t i = fAttrAxis+1; i<fShapeX.size(); i++) offset *= fShapesY[0][i];
      std::vector<size_t>out_offsets(fAttrAxis+2);
      out_offsets[fAttrAxis+1] = offset;
      for(int i = fAttrAxis; i>=0; i--)out_offsets[i] = fShapeX[i] * out_offsets[i+1];
      out << SP << "std::vector<size_t> out_offsets(" << ConvertShapeToString(out_offsets) <<");\n";
      size_t out_start = 0;
      out << SP << "size_t offset = 0, id_input = 0, itr = 0;\n";
      for (size_t i = 0; i < fNYs.size(); i++) {
        out << SP << "offset = 0, id_input = 0, itr = " << std::to_string(fAttrAxis) << ";\n";
        const size_t output_length = ConvertShapeToLength(fShapesY[i]);
        out << SP << "for (size_t id = 0; id < " << output_length << " ; id++){\n";
            out << SP << SP << "if(id_input == " << fShapesY[i][fAttrAxis]*offset << ") {\n";
            out << SP << SP << SP << "if(id == out_offsets[itr-1]-1) itr--;\n";
            out << SP << SP << SP << "offset += out_offsets[itr];\n";
            out << SP << SP << SP << "id_input = 0;\n";
            out << SP << SP << "}\n";
            out << SP << SP  << "tensor_" << fNYs[i] << "[id] = tensor_" << fNX << "[" << out_start << " + offset + id_input++];\n";
        out << SP << "}\n";
        out_start += fShapesY[i][fAttrAxis]*offset;
      }
      return out.str();
    }

   std::vector<std::string> GetStdLibs() { return { std::string("cmath") };}
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Split

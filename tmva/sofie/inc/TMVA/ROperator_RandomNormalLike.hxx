#ifndef TMVA_SOFIE_ROPERATOR_RandomNormalLike
#define TMVA_SOFIE_ROPERATOR_RandomNormalLike

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"  

#include <sstream>
#include <random>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_RandomNormalLike final : public ROperator
{

private:
    ETensorType output_type;
    std::mt19937 fRandomGen; // for seed gen
    int fAttrdtype = -1; // optional
    float fAttrMean = 0.0;
    float fAttrScale = 1.0;
    float fAttrSeed = 0; // optional
    std::string fNX;
    std::string fNY;
    std::vector<size_t> fShape;

public:
    ROperator_RandomNormalLike(){}
    ROperator_RandomNormalLike(int dtype, float mean, float scale, float seed, std::string nameX, std::string nameY):
        fAttrdtype(dtype), fAttrMean(mean), fAttrScale(scale), fAttrSeed(seed), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}
    ROperator_RandomNormalLike(float mean, float scale, float seed, std::string nameX, std::string nameY):
        fAttrMean(mean), fAttrScale(scale), fAttrSeed(seed), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}
    ROperator_RandomNormalLike(float mean, float scale, std::string nameX, std::string nameY):
        fAttrMean(mean), fAttrScale(scale), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}
    ROperator_RandomNormalLike(int dtype, float mean, float scale, std::string nameX, std::string nameY):
        fAttrdtype(dtype), fAttrMean(mean), fAttrScale(scale), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

    void Initialize(RModel& model){
        //input must be a graph input, or already initialized intermediate tensor
        model.AddNeededCustomHeader("random");
        if (model.CheckIfTensorAlreadyExist(fNX) == false){
            throw std::runtime_error("TMVA SOFIE RandomNormalLike Op Input Tensor is not found in model");
        }
        fShape = model.GetTensorShape(fNX);
        
        if (fAttrSeed == 0) {
            fAttrSeed = std::random_device{}();
        }
        output_type = fAttrdtype==-1 ? model.GetTensorType(fNX):static_cast<ETensorType>(fAttrdtype);
        model.AddIntermediateTensor(fNY, output_type, fShape);
    }

   std::string Generate(std::string OpName){
        OpName = "op_" + OpName;
        if (fShape.empty()) {
            throw std::runtime_error("TMVA SOFIE RandomNormalLike operator called to Generate without being initialized first");
        }
        std::stringstream out;
        size_t length = ConvertShapeToLength(fShape);
        out << "\n//------ RandomNormalLike\n";
        out << SP << "std::normal_distribution<" << ConvertTypeToString(output_type) << "> dist(" << fAttrMean << ", " << fAttrScale << ");\n";
        out << SP << "std::mt19937 generator;\n";  // random number generator
        out << SP << "generator.seed(" << fAttrSeed << ");\n";
        out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
        out << SP << SP << "tensor_" << fNY << "[id] = dist(generator);\n";  // generating random values
        out << SP << "}\n";
        return out.str();
    }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_RandomNormalLike

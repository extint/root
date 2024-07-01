#ifndef TMVA_SOFIE_ROperator_BasicBinary
#define TMVA_SOFIE_ROperator_BasicBinary

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum EBasicBinaryOperator { Add, Sub, Mul, Div, Pow };

template <typename T, EBasicBinaryOperator Op1>
struct BinaryOperatorTrait {};

template <typename T>
struct BinaryOperatorTrait<T, Add> {
   static const std::string Name() { return "Add"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " + " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Sub> {
   static const std::string Name() { return "Sub"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " - " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Mul> {
   static const std::string Name() { return "Mul"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " * " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Div> {
   static const std::string Name() { return "Div"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " / " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Pow> {
   static const std::string Name() { return "Pow"; }
   static std::string Op(const std::string & t1, const std::string t2) { return "std::pow(" + t1 + "," + t2 + ")"; }
};

template <typename T>
struct TensorType {};
template<>
struct TensorType<float> {
   static const std::string Name() { return "float"; }
};
template<>
struct TensorType<double> {
   static const std::string Name() { return "double"; }
};
template<>
struct TensorType<int64_t> {
   static const std::string Name() { return "int64_t"; }
};
template<>
struct TensorType<int32_t> {
   static const std::string Name() { return "int32_t"; }
};
template<>
struct TensorType<uint32_t> {
   static const std::string Name() { return "uint32_t"; }
};
template<>
struct TensorType<uint64_t> {
   static const std::string Name() { return "uint64_t"; }
};

template<typename T, EBasicBinaryOperator Op>
class ROperator_BasicBinary final : public ROperator{
private:

   std::string fNA;
   std::string fNB;
   std::string fNBroadcadstedA;
   std::string fNBroadcadstedB;
   std::string fNY;
   bool fIsDynamic = false;
   bool broadcast = false;

   std::vector<Dim> fShapeA;
   std::vector<Dim> fShapeB;
   std::vector<Dim> fShapeY;

public:
   ROperator_BasicBinary(){}
   ROperator_BasicBinary(std::string nameA, std::string nameB, std::string nameY):
      fNA(UTILITY::Clean_name(nameA)), fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY)){}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      // assume now inputs have same shape (no broadcasting)
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }

      std::vector<std::vector<Dim>> ShapeInference(std::vector<std::vector<Dim>> input)  {
      // assume now inputs have same shape (no broadcasting)
      auto ret = std::vector<std::vector<Dim>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNA)){
         throw std::runtime_error(std::string("TMVA SOFIE Binary Op Input Tensor ") + fNA + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNB)) {
         throw std::runtime_error(std::string("TMVA SOFIE Binary Op Input Tensor ") + fNB + "is not found in model");
      }
      if (model.IsDynamicTensor(fNA) || model.IsInputTensor(fNA) ) {
            fShapeA = model.GetDynamicTensorShape(fNA);
            fIsDynamic = true;
         }
      else {
         auto shapeA_int = model.GetTensorShape(fNA);
         fShapeA = ConvertShapeToDim(shapeA_int);
      }
      if (model.IsDynamicTensor(fNB) || model.IsInputTensor(fNB)) {
         fShapeB = model.GetDynamicTensorShape(fNB);
         fIsDynamic = true;
      }
      else {
         auto shapeB_int = model.GetTensorShape(fNB);
         fShapeB = ConvertShapeToDim(shapeB_int);
      }
      std::string dyn="nothing A\n";
      if(model.IsDynamicTensor(fNA))
       dyn= "dynamicT A\n";
      if(model.IsInputTensor(fNA))
       dyn= "inputT A\n";
      std::cout<<dyn;
      broadcast = !UTILITY::AreSameShape(fShapeA, fShapeB);
      {
         std::cout<<"A: ";
         for(auto i:fShapeA)std::cout<<i.GetVal()<<" ";
         std::cout<<"\nB: ";
         for(auto i:fShapeB)std::cout<<i.GetVal()<<" ";
         std::cout<<std::endl;      
         std::cout<<broadcast<<" broad\n";
      }

      if (broadcast) {
         if(!fIsDynamic){
            // Y is the common shape of A and B
            std::vector<size_t> ShapeA_int=model.GetTensorShape(fNA);
            std::vector<size_t> ShapeB_int=model.GetTensorShape(fNB);
            std::vector<size_t> ShapeY_int;
            // fShapeY = UTILITY::UnidirectionalBroadcastShape(ShapeA_int, ShapeB_int);
            ShapeY_int = UTILITY::UnidirectionalBroadcastShape(ShapeA_int, ShapeB_int);
            fShapeY = ConvertShapeToDim(ShapeY_int);
            std::cout<<"output\n";
            for(auto i:fShapeY)std::cout<<i.GetVal()<<" ";
               std::cout<<std::endl;
            bool broadcastA = !UTILITY::AreSameShape(ShapeA_int, ShapeY_int);
            bool broadcastB = !UTILITY::AreSameShape(ShapeB_int, ShapeY_int);
            // Broadcast A to Y
            if (broadcastA) {
               if (model.IsInitializedTensor(fNA)) {
                  auto data = model.GetInitializedTensorData(fNA);
                  std::shared_ptr<void> broadcastedData(
                     UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), ShapeA_int, ShapeY_int),
                     std::default_delete<T[]>());
                  // Update the data and the shape of A
                  model.UpdateInitializedTensor(fNA, model.GetTensorType(fNA), ShapeY_int, broadcastedData);
                  ShapeA_int = ShapeY_int;
                  fShapeB = ConvertShapeToDim(ShapeB_int); //???
               } else {
                  // Add an intermediate tensor for broadcasting A
                  fNBroadcadstedA = "Broadcasted" + fNA;
                  model.AddIntermediateTensor(fNBroadcadstedA, model.GetTensorType(fNA), ShapeY_int);
               }
            }
            // Broadcast B to Y
            if (broadcastB) {
               if (model.IsInitializedTensor(fNB)) {
                  auto data = model.GetInitializedTensorData(fNB);
                  std::shared_ptr<void> broadcastedData(
                     UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), ShapeB_int, ShapeY_int),
                     std::default_delete<T[]>());
                  // Update the data and the shape of B
                  model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), ShapeY_int, broadcastedData);
                  ShapeB_int = ShapeY_int;
                  fShapeB = ConvertShapeToDim(ShapeB_int); // ???
               } else {
                  // Add an intermediate tensor for broadcasting B
                  fNBroadcadstedB = "Broadcasted" + fNB;
                  model.AddIntermediateTensor(fNBroadcadstedB, model.GetTensorType(fNB), ShapeY_int);
               }
            }
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
         }
         else {
            // std::vector<Dim> bcastedShapeY=fShapeA; //need to figure out -> this is just a temporary shape, it will be updated after broadcasting in the init code
                                                    // at runtime
            // Add an intermediate tensor for broadcasting B
            fNBroadcadstedA = "Broadcasted" + fNA;
            model.AddIntermediateTensor(fNBroadcadstedA, model.GetTensorType(fNA), fShapeA);
            // Add an intermediate tensor for broadcasting B
            fNBroadcadstedB = "Broadcasted" + fNB;
            model.AddIntermediateTensor(fNBroadcadstedB, model.GetTensorType(fNB), fShapeB);
            // auto shapeA = ConvertDynamicShapeToString(fShapeA);
            // auto shapeB = ConvertDynamicShapeToString(fShapeB);
            // std::string nameA = fNA+"_shape";
            // std::string nameB = fNB+"_shape";
            // model.AddIntermediateTensor(nameA, model.GetTensorType(fNA), fShapeA);
            // model.AddIntermediateTensor(nameB, model.GetTensorType(fNB), fShapeB);
            fShapeY = fShapeA;                                        
            model.AddDynamicTensor(fNY, model.GetTensorType(fNA),fShapeY);
         }
      } else {
         fShapeY = fShapeA;
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
      }
   }

   // std::string GenerateInitCode() override {
   //    std::stringstream out;
   //    return out.str();
   // }

   std::string GenerateInitCode() {
      std::stringstream out;
      // generate initialization code for broadcasting of output tensor
      if (fIsDynamic && broadcast) {
         // parametric dynamic tensors
         auto shapeA = ConvertDynamicShapeToString(fShapeA);
         auto shapeB = ConvertDynamicShapeToString(fShapeB);
         out << SP << "std::vector<size_t> fShapeA = " << shapeA << ";\n";
         out << SP << "std::vector<size_t> fShapeB = " << shapeB << ";\n";
         out << SP << "std::vector<size_t> OutputShape = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcastShape(fShapeA, fShapeB);\n";
         out << SP << "std::size_t OutputLength=1;\n";
         out << SP << "for(auto i:OutputShape)\n";
         out << SP << SP << "OutputLength*=i;\n";   
         out << SP << "fTensor_" + fNY + ".resize(OutputLength);\n";


         out << SP << "bool broadcastA = !TMVA::Experimental::SOFIE::UTILITY::AreSameShape(fShapeA, OutputShape);\n";
         out << SP << "bool broadcastB = !TMVA::Experimental::SOFIE::UTILITY::AreSameShape(fShapeB, OutputShape);\n";

         out << SP << "if (broadcastA) {\n";
         out << SP << SP << "// resize intermediate tensor for broadcasting A\n";
         out << SP << SP << "fTensor_Broadcasted" + fNA + ".resize(OutputLength);\n";
         out << SP << SP << "tensor_Broadcasted" + fNA + " = fTensor_Broadcasted" + fNA + ".data();\n";
         out << SP << "}\n";
         out << SP << "if (broadcastB) {\n";
         out << SP << SP << "// resize intermediate tensor for broadcasting B\n";
         out << SP << SP << "fTensor_Broadcasted" + fNB + ".resize(OutputLength);\n";
         out << SP << SP << "tensor_Broadcasted" + fNB + " = fTensor_Broadcasted" + fNB + ".data();\n";
         out << SP << "}\n";
      }
      return out.str();
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Binary Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ " << BinaryOperatorTrait<T,Op>::Name() << "\n";
      // out<<"std::size_t length = TMVA::Experimental::SOFIE::UTILITY::ConvertShapeToLength("+ConvertDynamicShapeToString(fShapeY)+");\n";
      out << SP << "std::size_t OutputLength = fTensor_" + fNY + ".size();\n";
      auto shapeA = ConvertDynamicShapeToString(fShapeA);
      auto shapeB = ConvertDynamicShapeToString(fShapeB);
      out << SP << "std::vector<size_t> fShapeA = " << shapeA << ";\n";
      out << SP << "std::vector<size_t> fShapeB = " << shapeB << ";\n";
      out << SP << "std::vector<size_t> OutputShape = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcastShape(fShapeA, fShapeB);\n";
      std::string typeName = TensorType<T>::Name();
      // Broadcast A if it's uninitialized
      if (!fNBroadcadstedA.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNA << "\n";
         out << SP << "{\n";
         out << SP << SP << typeName << "* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNA << ", " << ConvertDynamicShapeToString(fShapeA) << ",OutputShape );\n";
         out << SP << SP << "std::copy(data, data + OutputLength, tensor_" << fNBroadcadstedA << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      // Broadcast B if it's uninitialized
      if (!fNBroadcadstedB.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNB << "\n";
         out << SP << "{\n";
         out << SP << SP << typeName << "* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNB << ", " << ConvertDynamicShapeToString(fShapeB) << ",OutputShape );\n";
         out << SP << SP << "std::copy(data, data + OutputLength, tensor_" << fNBroadcadstedB << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      const std::string& nameA = fNBroadcadstedA.empty()? fNA : fNBroadcadstedA;
      const std::string& nameB = fNBroadcadstedB.empty()? fNB : fNBroadcadstedB;
      out << SP << "for (size_t id = 0; id < OutputLength ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = "  << BinaryOperatorTrait<T,Op>::Op( "tensor_" + nameA + "[id]" , "tensor_" + nameB + "[id]") <<  " ;\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      if (Op == EBasicBinaryOperator::Pow) {
         return { std::string("cmath") };
      } else {
         return {};
      }
   }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_BasicBinary

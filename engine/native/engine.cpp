#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>

namespace py = pybind11;

class OnnxWrapper {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<std::string> input_node_names_alloc; // Para manter a memória das strings
    std::vector<std::string> output_node_names_alloc;

public:
    OnnxWrapper(const std::string& model_path, bool use_gpu) 
        : env(ORT_LOGGING_LEVEL_WARNING, "DubberEngine"), session(nullptr) {
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1); // Deixe o paralelismo para o batching
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Configuração de GPU (DirectML ou CUDA)
        if (use_gpu) {
            // Em produção, aqui você verifica qual Provider está disponível
            // Por simplicidade, tentamos CUDA primeiro, depois CPU fallback
            try {
                // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
                // Nota: Requer onnxruntime-gpu instalado e linkado.
                // Se der erro de linkagem, comente a linha acima para testar CPU-only primeiro.
            } catch (...) {
                std::cerr << "[WARN] Falha ao carregar GPU provider. Usando CPU." << std::endl;
            }
        }

        // Carrega o Modelo
        #ifdef _WIN32
            // Windows precisa de wstring para caminhos
            std::wstring w_model_path(model_path.begin(), model_path.end());
            session = Ort::Session(env, w_model_path.c_str(), session_options);
        #else
            session = Ort::Session(env, model_path.c_str(), session_options);
        #endif

        // Detecta nomes das entradas/saídas automaticamente (Introspecção)
        size_t num_input_nodes = session.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_node_names_alloc.push_back(name.get());
            input_node_names.push_back(input_node_names_alloc.back().c_str());
        }

        size_t num_output_nodes = session.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_node_names_alloc.push_back(name.get());
            output_node_names.push_back(output_node_names_alloc.back().c_str());
        }
    }

    // Função Genérica: Recebe Dicionário de Inputs (Nome -> NumpyArray)
    std::vector<py::array> Infer(const std::unordered_map<std::string, py::array>& inputs) {
        
        std::vector<Ort::Value> input_tensors;
        
        // 1. Converte Numpy -> Ort::Value (Tensores)
        for (const char* node_name : input_node_names) {
            if (inputs.find(node_name) == inputs.end()) {
                throw std::runtime_error(std::string("Missing input: ") + node_name);
            }

            const py::array& np_array = inputs.at(node_name);
            py::buffer_info info = np_array.request();

            // Descobre o tipo de dado e cria o Tensor adequado
            // Por simplicidade, assumimos int64 ou float32 (comuns em NLP)
            if (info.format == py::format_descriptor<int64_t>::format()) {
                input_tensors.push_back(CreateTensor<int64_t>(info));
            } else if (info.format == py::format_descriptor<float>::format()) {
                input_tensors.push_back(CreateTensor<float>(info));
            } else {
                 throw std::runtime_error("Unsupported tensor type provided from Python.");
            }
        }

        // 2. Executa Inferência
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_node_names.data(),
            output_node_names.size()
        );

        // 3. Converte Ort::Value -> Numpy (para devolver ao Python)
        std::vector<py::array> outputs;
        for (size_t i = 0; i < output_tensors.size(); i++) {
            outputs.push_back(TensorToNumpy(output_tensors[i]));
        }

        return outputs;
    }

private:
    // Helpers para conversão de memória (Zero-Copy quando possível)
    template <typename T>
    Ort::Value CreateTensor(const py::buffer_info& info) {
        std::vector<int64_t> shapes;
        for (auto s : info.shape) shapes.push_back(s);
        
        return Ort::Value::CreateTensor<T>(
            allocator.GetInfo(),
            reinterpret_cast<T*>(info.ptr),
            info.size,
            shapes.data(),
            shapes.size()
        );
    }

    py::array TensorToNumpy(Ort::Value& tensor) {
        auto type_info = tensor.GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();
        
        // Apenas exemplo para Float (Output comum de modelos)
        // Você pode expandir para int64 se o modelo retornar IDs
        if (type_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return py::array_t<float>(shape, tensor.GetTensorMutableData<float>());
        } else if (type_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return py::array_t<int64_t>(shape, tensor.GetTensorMutableData<int64_t>());
        }
        return py::array();
    }
};

PYBIND11_MODULE(dubber_engine, m) {
    py::class_<OnnxWrapper>(m, "OnnxWrapper")
        .def(py::init<const std::string&, bool>())
        .def("infer", &OnnxWrapper::Infer);
}
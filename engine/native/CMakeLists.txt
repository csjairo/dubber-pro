cmake_minimum_required(VERSION 3.10)
project(dubber_engine)

set(CMAKE_CXX_STANDARD 17)

# 1. Encontra Python e PyBind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# 2. Configura ONNX Runtime (Assumindo que você baixou os binários ou instalou via nuget/brew/apt)
# Ajuste este caminho para onde você extraiu o ONNX Runtime Headers/Libs
# Se estiver no Windows via Nuget, os caminhos podem variar.
# Exemplo simplificado para Windows:
set(ONNX_ROOT "C:/onnxruntime-win-x64-1.16.3") 
include_directories("${ONNX_ROOT}/include")
link_directories("${ONNX_ROOT}/lib")

# 3. Cria o módulo
pybind11_add_module(dubber_engine engine.cpp)

# 4. Linka bibliotecas (onnxruntime.lib no Windows, onnxruntime.so no Linux)
if(WIN32)
    target_link_libraries(dubber_engine PRIVATE "${ONNX_ROOT}/lib/onnxruntime.lib")
else()
    target_link_libraries(dubber_engine PRIVATE onnxruntime)
endif()

# Otimizações
if(MSVC)
    target_compile_options(dubber_engine PRIVATE /O2 /Ob2)
else()
    target_compile_options(dubber_engine PRIVATE -O3 -march=native)
endif()

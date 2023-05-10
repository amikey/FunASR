#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "paraformer.grpc.pb.h"
#include "paraformer-server.h"
#include "tclap/CmdLine.h"
#include "com-define.h"
#include "glog/logging.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using paraformer::Request;
using paraformer::Response;
using paraformer::ASR;

ASRServicer::ASRServicer(std::map<std::string, std::string>& model_path) {
    AsrHanlde=FunOfflineInit(model_path, 1);
    std::cout << "ASRServicer init" << std::endl;
    init_flag = 0;
}

void ASRServicer::clear_states(const std::string& user) {
    clear_buffers(user);
    clear_transcriptions(user);
}

void ASRServicer::clear_buffers(const std::string& user) {
    if (client_buffers.count(user)) {
        client_buffers.erase(user);
    }
}

void ASRServicer::clear_transcriptions(const std::string& user) {
    if (client_transcription.count(user)) {
        client_transcription.erase(user);
    }
}

void ASRServicer::disconnect(const std::string& user) {
    clear_states(user);
    std::cout << "Disconnecting user: " << user << std::endl;
}

grpc::Status ASRServicer::Recognize(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<Response, Request>* stream) {

    Request req;
    while (stream->Read(&req)) {
        if (req.isend()) {
            std::cout << "asr end" << std::endl;
            disconnect(req.user());
            Response res;
            res.set_sentence(
                R"({"success": true, "detail": "asr end"})"
            );
            res.set_user(req.user());
            res.set_action("terminate");
            res.set_language(req.language());
            stream->Write(res);
        } else if (req.speaking()) {
            if (req.audio_data().size() > 0) {
                auto& buf = client_buffers[req.user()];
                buf.insert(buf.end(), req.audio_data().begin(), req.audio_data().end());
            }
            Response res;
            res.set_sentence(
                R"({"success": true, "detail": "speaking"})"
            );
            res.set_user(req.user());
            res.set_action("speaking");
            res.set_language(req.language());
            stream->Write(res);
        } else if (!req.speaking()) {
            if (client_buffers.count(req.user()) == 0 && req.audio_data().size() == 0) {
                Response res;
                res.set_sentence(
                    R"({"success": true, "detail": "waiting_for_voice"})"
                );
                res.set_user(req.user());
                res.set_action("waiting");
                res.set_language(req.language());
                stream->Write(res);
            }else {
                auto begin_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                if (req.audio_data().size() > 0) {
                  auto& buf = client_buffers[req.user()];
                  buf.insert(buf.end(), req.audio_data().begin(), req.audio_data().end());
                }
                std::string tmp_data = this->client_buffers[req.user()];
                this->clear_states(req.user());

                Response res;
                res.set_sentence(
                    R"({"success": true, "detail": "decoding data: " + std::to_string(tmp_data.length()) + " bytes"})"
                );
                int data_len_int = tmp_data.length();
                std::string data_len = std::to_string(data_len_int);
                std::stringstream ss;
                ss << R"({"success": true, "detail": "decoding data: )" << data_len << R"( bytes")"  << R"("})";
                std::string result = ss.str();
                res.set_sentence(result);
                res.set_user(req.user());
                res.set_action("decoding");
                res.set_language(req.language());
                stream->Write(res);
                if (tmp_data.length() < 800) { //min input_len for asr model
                    auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    std::string delay_str = std::to_string(end_time - begin_time);
                    std::cout << "user: " << req.user() << " , delay(ms): " << delay_str << ", error: data_is_not_long_enough" << std::endl;
                    Response res;
                    std::stringstream ss;
                    std::string asr_result = "";
                    ss << R"({"success": true, "detail": "finish_sentence","server_delay_ms":)" << delay_str << R"(,"text":")" << asr_result << R"("})";
                    std::string result = ss.str();
                    res.set_sentence(result);
                    res.set_user(req.user());
                    res.set_action("finish");
                    res.set_language(req.language());
                    stream->Write(res);
                }
                else {
                    FUNASR_RESULT Result= FunOfflineRecogPCMBuffer(AsrHanlde, tmp_data.c_str(), data_len_int, 16000, RASR_NONE, NULL);
                    std::string asr_result = ((FUNASR_RECOG_RESULT*)Result)->msg;

                    auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    std::string delay_str = std::to_string(end_time - begin_time);

                    std::cout << "user: " << req.user() << " , delay(ms): " << delay_str << ", text: " << asr_result << std::endl;
                    Response res;
                    std::stringstream ss;
                    ss << R"({"success": true, "detail": "finish_sentence","server_delay_ms":)" << delay_str << R"(,"text":")" << asr_result << R"("})";
                    std::string result = ss.str();
                    res.set_sentence(result);
                    res.set_user(req.user());
                    res.set_action("finish");
                    res.set_language(req.language());

                    stream->Write(res);
                }
            }
        }else {
            Response res;
            res.set_sentence(
                R"({"success": false, "detail": "error, no condition matched! Unknown reason."})"
            );
            res.set_user(req.user());
            res.set_action("terminate");
            res.set_language(req.language());
            stream->Write(res);
        }
    }
    return Status::OK;
}

void RunServer(std::map<std::string, std::string>& model_path) {
    std::string port;
    try{
        port = model_path.at(PORT_ID);
    }catch(std::exception const &e){
        printf("Error when read port.\n");
        exit(0);
    }
    std::string server_address;
    server_address = "0.0.0.0:" + port;
    ASRServicer service(model_path);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

void GetValue(TCLAP::ValueArg<std::string>& value_arg, std::string key, std::map<std::string, std::string>& model_path)
{
    if (value_arg.isSet()){
        model_path.insert({key, value_arg.getValue()});
        LOG(INFO)<< key << " : " << value_arg.getValue();
    }
}

int main(int argc, char* argv[]) {

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("paraformer-server", ' ', "1.0");
    TCLAP::ValueArg<std::string>    model_dir("", MODEL_DIR, "the asr model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
    TCLAP::ValueArg<std::string>    quantize("", QUANTIZE, "false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "false", "string");
    TCLAP::ValueArg<std::string>    vad_dir("", VAD_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
    TCLAP::ValueArg<std::string>    vad_quant("", VAD_QUANT, "false (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "false", "string");
    TCLAP::ValueArg<std::string>    punc_dir("", PUNC_DIR, "the punc model path, which contains model.onnx, punc.yaml", false, "", "string");
    TCLAP::ValueArg<std::string>    punc_quant("", PUNC_QUANT, "false (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "false", "string");
    TCLAP::ValueArg<std::string>    port_id("", PORT_ID, "port id", true, "", "string");

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_quant);
    cmd.add(port_id);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);
    GetValue(port_id, PORT_ID, model_path);

    RunServer(model_path);
    return 0;
}
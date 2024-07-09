#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

std::string input_path;
std::string output_path;
llvm::LLVMContext* TheContext;
std::unique_ptr<llvm::Module> TheModule;
llvm::GlobalVariable* addInstCount;

void ParseIRSource(void);
void TraverseModule(void);
void PrintModule(void);
void AddGlobalVariable(void);
void AddPrintFunction(void);

int main(int argc , char** argv)
{
	if(argc < 3)
	{
		std::cout << "Usage: ./InsertInst <input_ir_file> <output_ir_file>" << std::endl;
		return -1;
	}
	input_path = std::string(argv[1]);
	output_path = std::string(argv[2]);

	// Read & Parse IR Source
	ParseIRSource();
	// Add Global Variable
	AddGlobalVariable();
	// Traverse TheModule
	TraverseModule();
	// Add Print Function
	AddPrintFunction();
	// Print TheModule to output_path
	PrintModule();

	return 0;
}

// Read & Parse IR Sources
//  Human-readable assembly(*.ll) or Bitcode(*.bc) format is required
void ParseIRSource(void)
{
	llvm::SMDiagnostic err;

	// Context
	TheContext = new llvm::LLVMContext();
	if( ! TheContext )
	{
		std::cerr << "Failed to allocated llvm::LLVMContext" << std::endl;
		exit( -1 );
	}

	// Module from IR Source
	TheModule = llvm::parseIRFile(input_path, err, *TheContext);
	if( ! TheModule )
	{
		std::cerr << "Failed to parse IR File : " << input_path << std::endl;
		exit( -1 );
	}
}

void AddGlobalVariable(void)
{
    llvm::Type* i32Type = llvm::Type::getInt32Ty(*TheContext);
    new llvm::GlobalVariable(
        *TheModule,
        i32Type,
        false,
        llvm::GlobalValue::CommonLinkage,
        llvm::ConstantInt::get(i32Type, 0),
        "add_inst_count"
    );
}


// Traverse Instructions in TheModule
void TraverseModule(void)
{
    llvm::IRBuilder<> Builder(*TheContext);
    llvm::Type* i32Type = llvm::Type::getInt32Ty(*TheContext);
    llvm::GlobalVariable* addInstCount = TheModule->getNamedGlobal("add_inst_count");

    for (auto &F : *TheModule)
    {
        for (auto &BB : F)
        {
            for (auto &I : BB)
            {
                if (llvm::isa<llvm::BinaryOperator>(I) && I.getOpcode() == llvm::Instruction::Add)
                {
                    Builder.SetInsertPoint(&I);
                    Builder.CreateStore(
                        Builder.CreateAdd(
                            Builder.CreateLoad(i32Type, addInstCount),
                            llvm::ConstantInt::get(i32Type, 1)
                        ),
                        addInstCount
                    );
                }
            }
        }
    }
}

void AddPrintFunction(void)
{
    llvm::IRBuilder<> Builder(*TheContext);
    llvm::Type* i32Type = llvm::Type::getInt32Ty(*TheContext);
    llvm::GlobalVariable* addInstCount = TheModule->getNamedGlobal("add_inst_count");

    // Declare printf function
    llvm::FunctionType* printfType = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(*TheContext),
        {llvm::Type::getInt8PtrTy(*TheContext)},
        true
    );
    llvm::Constant* printfFunc = TheModule->getOrInsertFunction("printf", printfType);

    // Create print function
    llvm::FunctionType* printFuncType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(*TheContext),
        false
    );
    llvm::Function* printFunc = llvm::Function::Create(
        printFuncType,
        llvm::Function::ExternalLinkage,
        "print_add_count",
        TheModule.get()
    );

    llvm::BasicBlock* entryBB = llvm::BasicBlock::Create(*TheContext, "entry", printFunc);
    Builder.SetInsertPoint(entryBB);

    // Create printf call
    llvm::Value* formatStr = Builder.CreateGlobalStringPtr("Number of add instructions: %d\n");
    llvm::Value* loadedCount = Builder.CreateLoad(i32Type, addInstCount);
    Builder.CreateCall(llvm::cast<llvm::Function>(printfFunc), {formatStr, loadedCount});

    Builder.CreateRetVoid();
}

// Print TheModule to output path in human-readable format
void PrintModule(void)
{
	std::error_code err;
	llvm::raw_fd_ostream raw_output( output_path, err, llvm::sys::fs::OpenFlags::F_None );

	if( raw_output.has_error() )
	{
		std::cerr << "Failed to open output file : " << output_path << std::endl;
		exit(-1);
	}

	TheModule->print(raw_output, NULL);
	raw_output.close();
}



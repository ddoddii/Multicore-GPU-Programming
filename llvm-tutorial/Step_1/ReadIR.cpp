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

llvm::LLVMContext* TheContext;
std::unique_ptr<llvm::Module> TheModule;

int main(int argc , char** argv)
{
	std::string input_path;
	llvm::SMDiagnostic err;

	if(argc < 2)
	{
		std::cout << "Usage: ./ReadIR <ir_file_path>" << std::endl;
		return -1;
	}
	input_path = std::string(argv[1]);

	// Context owns and manages the core data of LLVM infrastructure, including the type and constant tables.
	// new llvm::LLVMContext() allocate Context.
	TheContext = new llvm::LLVMContext();
	if( !TheContext )
	{
		std::cout<<"fail to allocate LLVMContext"<<std::endl;
		return -1;
	}

	// Module is the top level container of LLVM intermediate Representation(IR) objects.
	// Module take as input BitCode File and parse IR format.
	TheModule = llvm::parseIRFile(input_path, err, *TheContext);
	if( !TheModule )
	{
		std::cout << "Cannot open the file : "<< input_path << std::endl;
		return -1;
	}

	// Print Module Name
	std::cout << "Success reading & parsing the IR file." << std::endl;
	std::cout << "The module name is \"" << TheModule->getName().str() << "\"" << std::endl;
	std::cout << std::endl;

	return 0;
}

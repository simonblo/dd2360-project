## Compile

- The project is developed on Windows 11 with Visual Studio 2022.
- No required dependencies except NVIDIA CUDA Toolkit 12.3 installation.
- Choose debug or release configuration and then build the solution (Build > Build Solution).

## Execute

- To execute through Visual Studio, first add the input file (Project > Properties > Debugging and add command argument $(ProjectDir)inputfiles/GEM_2D.inp) and then execute the application (Debug > Start Without Debugging).
- To execute through Command Prompt, first navigate to $(ProjectDir) and then execute the application (../x64/Release/Project.exe inputfiles/GEM_2D.inp).

## Validate

- Output is stored in the folder $(ProjectDir)/data which must be created manually.
- To generate reference output, call the function mover_PC_cpu in sputniPIC.cpp, then compile and execute.
- Output from mover_PC_gpu can be validated by comparison with reference output.
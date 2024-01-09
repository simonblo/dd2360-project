
compile

- platform, developed on windows 11 with visual studio 2022
- dependencies, no dependencies except installed cuda (NVIDIA CUDA Toolkit 12.3)
- choose debug or release in solution configuration and then build the solution (Build -> Build solution)

execute

- to execute through visual studio (Debug -> Start without debugging) (also add input file in properties)
- to execute through command prompt, navigate to $(ProjectDir) etc etc

validate

- where the output is stored (include that the data folder must be created manually)
- reference output can be obtained by using the function mover_PC_cpu and rebuilding
- the output from mover_PC_gpu can be validated by comparison with reference output

# [TODO]

compile

- platform
- dependencies
- choose debug or release

execute

- to execute through visual studio (Debug -> Start without debugging) (also add input file in properties)
- to execute through command prompt, navigate to $(ProjectDir) etc etc

validate

- where the output is stored (include that the data folder must be created manually)
- reference output can be obtained by using the function mover_PC_cpu and rebuilding
- output from mover_PC_gpu can be validated by comparison with reference output

# [TODO]
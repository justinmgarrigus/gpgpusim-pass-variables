# gpgpusim-pass-variables
Pass variables between the simulator and the source files being simulated.

# Simulator
The simulator contains some variable (e.g., `gpgpusim_test_var`) that should be set *from* the simulated 
code. This could represent something like convolutional parameters; the simulated code may want to pass 
matrix dimensions and pointers to the simulator. We can't do this through the config file because pointers 
do not have static values, so we have to set it at runtime. We can easily do this through files.

This code example edits the `gpgpu_sim::launch` method inside of `gpu-sim.cc`. This method is run before 
the kernel is executed. The two functions `read_extern_values` and `write_extern_values` read and write 
values to a file named `EXTERN_VARS.temp`.

```
// Reads specific values from the file. If the file is not found, exits the
// program.
void read_extern_values(int* gpgpusim_test_var) {
    FILE *extern_vars_file = fopen("EXTERN_VARS.temp", "r"); 
    assert(extern_vars_file != NULL); 
    fscanf(extern_vars_file, "gpgpusim_test_var=%d\n", gpgpusim_test_var); 
    fclose(extern_vars_file);
    printf("UNT (SIM): value received, %d\n", *gpgpusim_test_var);
}

// Writes specific values to the file.
void write_extern_values(int gpgpusim_test_var) {
    FILE *extern_vars_file = fopen("EXTERN_VARS.temp", "w"); 
    fprintf(extern_vars_file, "gpgpusim_test_var=%d\n", gpgpusim_test_var); 
    fclose(extern_vars_file); 
    printf("UNT (SIM): value written, %d\n", gpgpusim_test_var);
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  printf("UNT (SIM): Kernel about to launch.\n"); 
  
  // Read the values from the file.
  int gpgpusim_test_var; 
  read_extern_values(&gpgpusim_test_var); 
  
  // Change value. 
  gpgpusim_test_var++;
  
  // Rewrite values to file again.
  write_extern_values(gpgpusim_test_var);

  // The rest of the code in "gpgpu_sim::launch" goes here... 
```

# Simulated Code
This code is what the user wants to simulate with the simulator. Continuing our previous example, this could
be a matrix-multiplication kernel. Before and after we run the kernel, we can read and write the values of 
the variables.

```
#include <stdio.h> 

__global__ void kernel() { }

// Reads specific values from the file, assigning default values when the file
// does not exist.
void read_extern_values(int* gpgpusim_test_var) {
    FILE *extern_vars_file = fopen("EXTERN_VARS.temp", "r");
    if (extern_vars_file == NULL) {
        *gpgpusim_test_var = 0; 
    }
    else {
        fscanf(extern_vars_file, "gpgpusim_test_var=%d\n", gpgpusim_test_var); 
        fclose(extern_vars_file);
    }  

    printf("UNT (SRC): value received, %d\n", *gpgpusim_test_var);
}

// Writes specific values to the file.
void write_extern_values(int gpgpusim_test_var) {
    FILE *extern_vars_file = fopen("EXTERN_VARS.temp", "w"); 
    fprintf(extern_vars_file, "gpgpusim_test_var=%d\n", gpgpusim_test_var);
    printf("UNT (SRC): value written, %d\n", gpgpusim_test_var);
    fclose(extern_vars_file);
}

int main() {
    printf("UNT (SRC): main program launched\n"); 
    int gpgpusim_test_var;

    // Delete the initial file, if it exists. 
    remove("EXTERN_VARS.temp"); 
    
    // Repeat this two times (for demonstration purposes).
    for (int i = 0; i < 2; i++) {

        // Read values from file. 
        read_extern_values(&gpgpusim_test_var);
    
        // Change value. 
        gpgpusim_test_var++; 

        // Rewrite values to file before loading them again from the simulator.
        write_extern_values(gpgpusim_test_var); 

        // Run the simulator on a kernel.
        printf("UNT (SRC): executing kernel.\n"); 
        kernel<<<1,1>>>(); 
        cudaDeviceSynchronize();
        printf("UNT (SRC): finished executing kernel.\n"); 
    
   } 

   // Read final values. 
   read_extern_values(&gpgpusim_test_var);
}
```

# Compile and Test
First compile the simulator (e.g., `make -j8`) and then compile the source file (if you named it 
`source.cc` then run `nvcc source.cc --cudart shared`). Run the project with the simulator (`./a.out`). 
In the example above there are different print statements added to see what is happening; running the 
above with `./a.out | grep "UNT"` gives this output:

```
UNT (SRC): main program launched
UNT (SRC): value received, 0
UNT (SRC): value written, 1
UNT (SRC): executing kernel.
UNT (SIM): Kernel about to launch.
UNT (SIM): value received, 1
UNT (SIM): value written, 2
UNT (SRC): finished executing kernel.
UNT (SRC): value received, 2
UNT (SRC): value written, 3
UNT (SRC): executing kernel.
UNT (SIM): Kernel about to launch.
UNT (SIM): value received, 3
UNT (SIM): value written, 4
UNT (SRC): finished executing kernel.
UNT (SRC): value received, 4
```

The simulator (SIM) and the simulated code (SRC) alternate between reading and writing values. This can
be used to pass more complicated data types like pointers, arrays, structs, etc., which is not possible 
through the simulator configuration file (`gpgpusim.config`). 

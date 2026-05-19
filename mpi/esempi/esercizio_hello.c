#include <stdio.h>
#include <mpi.h>

int main (int argc, char** argv) {

    int rank, size, len;



    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char process_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(process_name, &len);
    printf("%s\n", process_name);

    if(rank == 0) {
        printf("Hello world from the master process\n");
    } else {
        printf("Hello world from the slave process\n");
    }

    printf("%d", rank);

    MPI_Finalize();

    return 0;
}
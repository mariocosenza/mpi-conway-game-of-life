#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

uint32_t SEED = 1234;
int PATTERN = 0; 
uint8_t READ = 0;

void write_matrix_to_file_fast(uint32_t M, uint32_t N) {
    char filename[256];
    sprintf(filename, "matrix_%ux%u_seed%u_pattern%d.bin", M, N, SEED, PATTERN);
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Errore nell'apertura del file");
        exit(EXIT_FAILURE);
    }

    uint8_t *row_buffer = calloc(N, sizeof(uint8_t));
    if (!row_buffer) {
        printf("Errore di allocazione memoria\n");
        exit(EXIT_FAILURE);
    }

    if (PATTERN == 0) {
        srand(SEED);
    }
    else if (PATTERN == 4) {
        printf("Inserisci la matrice %u x %u (0 o 1):\n", M, N);
        printf("Esempio:\n");
        for (uint32_t r = 0; r < 4; r++) {
            for (uint32_t c = 0; c < 4; c++) {
                printf("%c ", (r == 1 && c == 2) || (r == 2 && c >= 1 && c <= 3) ? '1' : '0');
            }
            printf("\n");
        }
        printf("----------------\n");
    }
     


    for (uint32_t r = 0; r < M; r++) {
        if (PATTERN != 0) {
            memset(row_buffer, 0, N * sizeof(uint8_t));
        }

        if (PATTERN == 0) {
            for (uint32_t c = 0; c < N; c++) {
                row_buffer[c] = (rand() % 2 == 0);
            }
        } else if (PATTERN == 1 && r < 3 && M >= 3 && N >= 3) {
            // Glider top left
            if (r == 0) { row_buffer[1] = 1; }
            else if (r == 1) { row_buffer[2] = 1; }
            else if (r == 2) { row_buffer[0] = 1; row_buffer[1] = 1; row_buffer[2] = 1; }
        } else if (PATTERN == 2 && M >= 3 && N >= 3) {
            // Blinker center
            if (r == M / 2) { row_buffer[N / 2 - 1] = 1; row_buffer[N / 2] = 1; row_buffer[N / 2 + 1] = 1; }
        } else if (PATTERN == 3 && M >= 2 && N >= 2) {
            // Block center
            if (r == M / 2 || r == M / 2 + 1) { row_buffer[N / 2] = 1; row_buffer[N / 2 + 1] = 1; }
        } else if (PATTERN == 4) {
            for (uint32_t c = 0; c < N; c++) {
                scanf("%hhu", &row_buffer[c]);
                if (row_buffer[c] != 0 && row_buffer[c] != 1) {
                    fprintf(stderr, "Input non valido alla riga %u, colonna %u: %hhu\n", r, c, row_buffer[c]);
                    free(row_buffer);
                    fclose(fp);
                    exit(EXIT_FAILURE);
                }
            }
        }

        fwrite(row_buffer, sizeof(uint8_t), N, fp);
    }

    free(row_buffer);
    fclose(fp);
    
    const char* pattern_names[] = {"Random", "Glider", "Blinker", "Block", "Custom"};
    printf("Matrice %u x %u scritta su %s (Pattern: %s)\n", M, N, filename, pattern_names[PATTERN]);
}

void parse_args(int argc, char **argv, uint32_t *M, uint32_t *N) {
    int opt;
    while ((opt = getopt(argc, argv, "N:M:S:P:")) != -1) {
        switch (opt) {
            case 'N': *N = (uint32_t)atoi(optarg); break;
            case 'M': *M = (uint32_t)atoi(optarg); break;
            case 'S': SEED = (uint32_t)atoi(optarg); break;
            case 'P': PATTERN = atoi(optarg); break;
            case '?':
            default:
                fprintf(stderr, "Usage: %s [-M rows] [-N cols] [-S seed] [-P pattern] [-R]\n", argv[0]);
                fprintf(stderr, "  -M rows    Number of rows (default: 100)\n");
                fprintf(stderr, "  -N cols    Number of columns (default: 100)\n");
                fprintf(stderr, "  -S seed    Random seed for pattern 0 (default: 1234)\n");
                fprintf(stderr, "  -P pattern Pattern type (0=random, 1=glider, 2=blinker, 3=block, 4=custom) (default: 0)\n");
                exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char **argv) {
    uint32_t M = 100;
    uint32_t N = 100;
    parse_args(argc, argv, &M, &N);
    
  
    write_matrix_to_file_fast(M, N);
    

    return 0;
}
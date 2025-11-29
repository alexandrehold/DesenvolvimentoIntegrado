#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <time.h>
#include<math.h>

/* ------------------------------ */
/* Função para carregar CSV       */
/* ------------------------------ */
double* load_csv(const char* filename, int rows, int cols)
{
    FILE* f = fopen(filename, "r");
    if(!f){
        printf("Erro ao abrir %s\n", filename);
        return NULL;
    }

    double* data = (double*)malloc(rows * cols * sizeof(double));

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            if(fscanf(f, "%lf,", &data[i*cols + j]) != 1){
                fclose(f);
                printf("Erro ao ler CSV: %s\n", filename);
                return NULL;
            }
        }
    }

    fclose(f);
    return data;
}

/* ------------------------------ */
/* CGNR Algorithm                 */
/* ------------------------------ */
void cgnr(double* H, double* b, double* x,
          int M, int N, int max_iter, double tol)
{
    double *r  = calloc(M, sizeof(double));
    double *p  = calloc(N, sizeof(double));
    double *Hp = calloc(M, sizeof(double));

    // r = b - Hx
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N,
                1.0, H, N, x, 1, 0.0, r, 1);
    for(int i=0;i<M;i++) r[i] = b[i] - r[i];

    // p = Hᵀ r
    cblas_dgemv(CblasRowMajor, CblasTrans, M, N,
                1.0, H, N, r, 1, 0.0, p, 1);

    double rr_old = cblas_ddot(N, p, 1, p, 1);

    for(int it=0; it < max_iter; it++)
    {
        // Hp = H p
        cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N,
                    1.0, H, N, p, 1, 0.0, Hp, 1);

        double alpha = rr_old / cblas_ddot(M, Hp, 1, Hp, 1);

        // x = x + alpha p
        cblas_daxpy(N, alpha, p, 1, x, 1);

        // r = r - alpha Hp
        cblas_daxpy(M, -alpha, Hp, 1, r, 1);

        // compute new q = Hᵀ r
        double *q = calloc(N, sizeof(double));
        cblas_dgemv(CblasRowMajor, CblasTrans, M, N,
                    1.0, H, N, r, 1, 0.0, q, 1);

        double rr_new = cblas_ddot(N, q, 1, q, 1);

        if(sqrt(rr_new) < tol){
            printf("Convergiu na iteração %d\n", it);
            free(q);
            break;
        }

        double beta = rr_new / rr_old;

        // p = q + beta p
        for(int i=0; i<N; i++) p[i] = q[i] + beta * p[i];

        rr_old = rr_new;
        free(q);
    }

    free(r);
    free(p);
    free(Hp);
}

/* ------------------------------ */
/* MAIN                           */
/* ------------------------------ */
int main()
{
    /* Alterável para qualquer tamanho */
    int M = 27904;   // número de linhas da matriz H
    int N = 900;     // número de colunas da matriz H (30×30)

    printf("Carregando matriz H...\n");
    double* H = load_csv("h.csv", M, N);

    printf("Carregando sinal b...\n");
    double* b = load_csv("sinal.csv", M, 1);

    if(!H || !b){
        printf("Erro ao carregar arquivos.\n");
        return 1;
    }

    double* x = calloc(N, sizeof(double));

    clock_t t0 = clock();

    cgnr(H, b, x, M, N, 200, 1e-12);

    clock_t t1 = clock();

    double time_sec = (double)(t1 - t0)/CLOCKS_PER_SEC;

    printf("\nTempo total = %.4f segundos\n", time_sec);

    /* Salva a imagem reconstruída */
    FILE* fout = fopen("reconstruida.csv", "w");
    for(int i=0;i<N;i++)
        fprintf(fout, "%lf\n", x[i]);
    fclose(fout);

    printf("Imagem salva em reconstruida.csv\n");

    free(H);
    free(b);
    free(x);
    return 0;
}

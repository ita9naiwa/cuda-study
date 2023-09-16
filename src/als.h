void _cuALS_iter(int *indices, int *indptr, float* data, int nnz,
                    float *X, float *Y,
                    int n_users, int n_items,
                    int d, float reg);

int load_matrix_to_cuda_memory(int **dev_indices, int **dev_indptr, float** dev_data,
                                   int *indices, int *indptr, float *data,
                                   int nnz, int n_users);

int load_factors_to_cuda_memory(float **dev_X, float **dev_Y, float *X, float *Y,
                                int n_users, int n_items, int d );
int finalize(float *X, float *Y, float *dev_X, float *dev_Y, int *u_indices, int *u_indptr, float *u_data, int *i_indices, int *i_indptr, float *i_data, int n_users, int n_items, int d);


void _cuALS_iter2(int *dev_indices, int *dev_indptr, float *dev_data, int nnz,
 float *X, float *Y, int n_users, int n_items, int d, float reg);
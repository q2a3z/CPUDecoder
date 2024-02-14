#include "stdafx.h"

#include <cv.h>
#include <highgui.h>
#define OUTPUT_IMG 1
#define ERROR_CONTEXT 0
#define SIGN_FILP 0
#define REMAPPING 0
#define ADAPTIVE_ARITHMETIC_CODING 1
#define TEST_ONE 1
#define AllLSR 1
#define contextLSR 1

#define Pixel_Shift 128

#define FILE_PATH "C:\\img\\out\\"
#define FILE_OUT_PATH "C:\\img\\out\\decode\\"
#define Pixel_Byte 1
//Traning Area
#define traning_area_left  6				//_____7*6_______6*7____
#define traning_area_right 6			//|			         |		              |
#define traning_area_top 6				//|			         |			          |
#define traning_area_last 6				//|			         |__________|
#define traning_area_size 84			//|__________|x
#define PredictorOrderSetting 6
//#define MAX(a,b) ( ( a > b) ? a : b )
//#define MIN(a,b) ( ( a < b) ? a : b )

//**********************Least Square Resolve****************************//
#define NR_END 1
#define FREE_ARG char*
#define MAX_nn_size 15 //cholesky

static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
static float maxarg1, maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
(maxarg1) : (maxarg2))

#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ?\
(minarg1) : (minarg2))

#define LMAX(a,b) (lmaxarg1=(a),lmaxarg2=(b),(lmaxarg1) > (lmaxarg2) ?\
(lmaxarg1) : (lmaxarg2))

#define LMIN(a,b) (lminarg1=(a),lminarg2=(b),(lminarg1) < (lminarg2) ?\
(lminarg1) : (lminarg2))

#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
(imaxarg1) : (imaxarg2))
static int iminarg1, iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
(iminarg1) : (iminarg2))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
//**********************Least Square Resolve****************************//

//------------------Arithmetic Coding-------------------------------//
#define NUMLOOPS 10000
#define ADAPT 1
#define FILE1 "foo"
#define MASK1 254
#define NSYM1 256

#define Code_value_bits 16
#define Top_value (((long)1<<Code_value_bits)-1)
#define First_qtr (Top_value/4+1)
#define Half	  (2*First_qtr)
#define Third_qtr (3*First_qtr)
#define Max_frequency 16383

typedef struct {
	FILE *fp;
	long low;
	long high;
	long fbits;
	int buffer;
	int bits_to_go;
	long total_bits;
} ac_encoder;

typedef struct {
	FILE *fp;
	long value;
	long low;
	long high;
	int buffer;
	int bits_to_go;
	int garbage_bits;
} ac_decoder;

typedef struct {
	int nsym;
	int *freq;
	int *cfreq;
	int adapt;
} ac_model;
//----------------------------Arithmetic Coding-------------------------------//

//--------------------------Entropy--------------------------------//
#define MAXLEN 100 //maximum string length
#define HISSIZE 1024 //maximum string length
double Log2(double n)
{
	// log(n)/log(2) is log2.  
	return log(n) / log(2);
}
int makehist(int *S, int *hist, int len){
	int wherechar[HISSIZE];
	int i, histlen;
	histlen = 0;
	for (i = 0; i<HISSIZE; i++)wherechar[i] = -1;
	for (i = 0; i<len; i++){
		if (wherechar[(int)(S[i] + 256)] == -1){
			wherechar[(int)(S[i] + 256)] = histlen;
			histlen++;
		}
		hist[wherechar[(int)(S[i] + 256)]]++;
	}
	return histlen;
}
double entropy(int *hist, int histlen, int len){
	int i;
	double H;
	H = 0;
	for (i = 0; i<histlen; i++){
		H -= (double)hist[i] / len*Log2((double)hist[i] / len);
	}
	return H;
}
//--------------------------Entropy--------------------------------//

using namespace cv;

unsigned int Height, Width;
int x, n, w, ne, nw, nn, ww, nne;//Neiber Pixel
int predicted; //¹w´ú¼v¹³­Èxp
int predictor_order;//¹w´ú¶¥¼Æ
float a16[12] = { 1 };//33311
int delta = 0;
int Count = 0;
//int N[8] = { 38, 78, 133, 256 };
//int N[8] = { 256, 256, 256, 256 };
int N[8] = { 7, 11, 20, 24, 34, 46, 103, 256 };//EDP N
/*
int N[8] = { 19, 27, 39, 54, 62, 81, 106, 128 };
int nN[8] = { 16, 23, 37, 52, 66, 85, 108, 128 };
*/
double q[8] = { 5, 10, 15, 25, 42, 60, 85, 140 };//EDP
int contextLSR_SUM[1024] = { 0 }, contextLSR_count[1024] = { 0 };
int contextMED_SUM[1024] = { 0 }, contextMED_count[1024] = { 0 };
int S[1024] = { 0 }, context_count[1024] = { 0 };
char *FILEName;

unsigned int choleskyCount = 0, svdCount = 0;//Cholesky & SVD Counter
double compress_time, normal_equation_time, Tran_time;
clock_t compress_start, compress_start2, compress_end, Tran_start, Tran_end;

ac_decoder acd;
ac_model acm1, acm2, acm3, acm4, acm5, acm6, acm7, acm8;

/******************************************µ{¦¡¶}©l*********************************************/

//3 String Combin
char *FileName(char *filepath, char *filename, char *filetype) {
	// ­pºâ©Ò»Ýªº°}¦Cªø«×  
	int length = strlen(filepath) + strlen(filename) + strlen(filetype) + 1;
	// ²£¥Í·sªº°}¦CªÅ¶¡  
	char *result = (char*)malloc(sizeof(char) * length);
	// ½Æ»s²Ä¤@­Ó¦r¦ê¦Ü·sªº°}¦CªÅ¶¡  
	strcpy(result, filepath);
	// ¦ê±µ²Ä¤G­Ó¦r¦ê¦Ü·sªº°}¦CªÅ¶¡  
	strcat(result, filename);
	strcat(result, filetype);
	return result;
}
//4 String Combin
char *OutFileName(char *filepath, char *filename, char *times, char *filetype) {
	// ­pºâ©Ò»Ýªº°}¦Cªø«×  
	int length = strlen(filepath) + strlen(filename) + strlen(times) + strlen(filetype) + 1;
	// ²£¥Í·sªº°}¦CªÅ¶¡  
	char *result = (char*)malloc(sizeof(char) * length);
	// ½Æ»s²Ä¤@­Ó¦r¦ê¦Ü·sªº°}¦CªÅ¶¡  
	strcpy(result, filepath);
	// ¦ê±µ²Ä¤G­Ó¦r¦ê¦Ü·sªº°}¦CªÅ¶¡  
	strcat(result, filename);
	strcat(result, times);
	strcat(result, filetype);
	return result;
}
//ReadImage OpenCV RawData -> Source (unsigned char -> int)
void ReadImage(unsigned char* rawdata, int* SourceImage){
	for (unsigned int w = 0; w < Width; w++){
		for (unsigned int h = 0; h < Height; h++){
			unsigned int x = w * Height + h;
			*(SourceImage + x) = *(rawdata + x);
		}
	}
}
//Write Image Source -> OpenCV RawData (int ->unsigned char)
void WriteImage(int* OutImage, unsigned char* Outrawdata){
	for (unsigned int w = 0; w < Width; w++){
		for (unsigned int h = 0; h < Height; h++){
			unsigned int x = w * Height + h;
			//if (*(OutImage + x) > 128 || *(OutImage + x) < -128)
			//	*(Outrawdata + x) = Pixel_Shift;
			//else
			*(Outrawdata + x) = *(OutImage + x);
		}
	}
}
//WriteData To Txt
void writedata(char *data, char *filename){
	FILE *fp;
	char *c;
	c = (char *)malloc(sizeof(char) * 2);
	sprintf(c, "%d", predictor_order);
	if ((fp = fopen(OutFileName(FILE_OUT_PATH, filename, c, ".txt"), "w")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char) * 8, (Width)*(Height)*(Pixel_Byte), fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}
//WriteTime
void writetime(char *data, char *filename){
	FILE *fp;
	if ((fp = fopen(FileName(FILE_OUT_PATH, filename, "time.txt"), "a")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), 10, fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}
//WriteCount
void writeCounttimes(char *data, char *filename){
	FILE *fp;
	if ((fp = fopen(FileName(FILE_OUT_PATH, filename, "SVD.txt"), "a")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), 15, fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}
//Write Entropy
void writeEntropy(char *data, char *filename){
	FILE *fp;
	if ((fp = fopen(FileName(FILE_OUT_PATH, filename, "Entropy.txt"), "a")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), 8, fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}


//******************************ºâ³N½s½X***************************************
static void output_bit(ac_encoder *, int);
static void bit_plus_follow(ac_encoder *, int);
static int input_bit(ac_decoder *);
static void update_model(ac_model *, int);

#define error(m)                                           \
do  {                                                      \
  fflush (stdout);                                         \
  fprintf (stderr, "%s:%d: error: ", __FILE__, __LINE__);  \
  fprintf (stderr, m);                                     \
  fprintf (stderr, "\n");                                  \
  exit (1);                                                \
}  while (0)

#define check(b,m)                                         \
do  {                                                      \
  if (b)                                                   \
    error (m);                                             \
}  while (0)

static void
output_bit(ac_encoder *ace, int bit)
{
	ace->buffer >>= 1;
	if (bit)
		ace->buffer |= 0x80;
	ace->bits_to_go -= 1;
	ace->total_bits += 1;
	if (ace->bits_to_go == 0)  {
		if (ace->fp)
			putc(ace->buffer, ace->fp);
		ace->bits_to_go = 8;
	}
	return;
}

static void
bit_plus_follow(ac_encoder *ace, int bit)
{
	output_bit(ace, bit);
	while (ace->fbits > 0)  {
		output_bit(ace, !bit);
		ace->fbits -= 1;
	}
	return;
}

static int
input_bit(ac_decoder *acd)
{
	int t;

	if (acd->bits_to_go == 0)  {
		acd->buffer = getc(acd->fp);
		if (acd->buffer == EOF)  {
			acd->garbage_bits += 1;
			if (acd->garbage_bits>Code_value_bits - 2)
				error("arithmetic decoder bad input file");
		}
		acd->bits_to_go = 8;
	}

	t = acd->buffer & 1;
	acd->buffer >>= 1;
	acd->bits_to_go -= 1;

	return t;
}

static void
update_model(ac_model *acm, int sym)
{
	int i;

	if (acm->cfreq[0] == Max_frequency)  {
		int cum = 0;
		acm->cfreq[acm->nsym] = 0;
		for (i = acm->nsym - 1; i >= 0; i--)  {
			acm->freq[i] = (acm->freq[i] + 1) / 2;
			cum += acm->freq[i];
			acm->cfreq[i] = cum;
		}
	}

	acm->freq[sym] += 1;
	for (i = sym; i >= 0; i--)
		acm->cfreq[i] += 1;

	return;
}

void
ac_encoder_init(ac_encoder *ace, const char *fn)
{

	if (fn)  {
		ace->fp = fopen(fn, "wb"); /* open in binary mode */
		check(!ace->fp, "arithmetic encoder could not open file");
	}
	else  {
		ace->fp = NULL;
	}

	ace->bits_to_go = 8;

	ace->low = 0;
	ace->high = Top_value;
	ace->fbits = 0;
	ace->buffer = 0;

	ace->total_bits = 0;

	return;
}

void
ac_encoder_done(ac_encoder *ace)
{
	ace->fbits += 1;
	if (ace->low < First_qtr)
		bit_plus_follow(ace, 0);
	else
		bit_plus_follow(ace, 1);
	if (ace->fp)  {
		putc(ace->buffer >> ace->bits_to_go, ace->fp);
		fclose(ace->fp);
	}

	return;
}

void
ac_decoder_init(ac_decoder *acd, const char *fn)
{
	int i;

	acd->fp = fopen(fn, "rb"); /* open in binary mode */
	fseek(acd->fp, 6, SEEK_SET);
	check(!acd->fp, "arithmetic decoder could not open file");

	acd->bits_to_go = 0;
	acd->garbage_bits = 0;

	acd->value = 0;
	for (i = 1; i <= Code_value_bits; i++)  {
		acd->value = 2 * acd->value + input_bit(acd);
	}
	acd->low = 0;
	acd->high = Top_value;

	return;
}

void
ac_decoder_done(ac_decoder *acd)
{
	fclose(acd->fp);

	return;
}

void
ac_model_init(ac_model *acm, int nsym, int *ifreq, int adapt)
{
	int i;

	acm->nsym = nsym;
	acm->freq = (int *)(void *)calloc(nsym, sizeof(int));
	check(!acm->freq, "arithmetic coder model allocation failure");
	acm->cfreq = (int *)(void *)calloc(nsym + 1, sizeof(int));
	check(!acm->cfreq, "arithmetic coder model allocation failure");
	acm->adapt = adapt;
	//if ifreq(initial frequency) is defined, use the initial frequency
	//else default is that every symbol has the same frequency
	if (ifreq)  {
		acm->cfreq[acm->nsym] = 0;
		for (i = acm->nsym - 1; i >= 0; i--)  {
			acm->freq[i] = ifreq[i];
			acm->cfreq[i] = acm->cfreq[i + 1] + acm->freq[i];
		}
		if (acm->cfreq[0] > Max_frequency)
			error("arithmetic coder model max frequency exceeded");
	}
	else  {
		for (i = 0; i<acm->nsym; i++) {
			acm->freq[i] = 1;
			acm->cfreq[i] = acm->nsym - i;
		}
		acm->cfreq[acm->nsym] = 0;
	}

	return;
}

void
ac_model_done(ac_model *acm)
{
	acm->nsym = 0;
	free(acm->freq);
	acm->freq = NULL;
	free(acm->cfreq);
	acm->cfreq = NULL;

	return;
}

long
ac_encoder_bits(ac_encoder *ace)
{
	return ace->total_bits;
}

void
ac_encode_symbol(ac_encoder *ace, ac_model *acm, int sym)
{
	long range;

	check(sym<0 || sym >= acm->nsym, "symbol out of range");

	range = (long)(ace->high - ace->low) + 1;
	ace->high = ace->low + (range*acm->cfreq[sym]) / acm->cfreq[0] - 1;
	ace->low = ace->low + (range*acm->cfreq[sym + 1]) / acm->cfreq[0];

	for (;;)  {
		if (ace->high<Half)  {
			bit_plus_follow(ace, 0);
		}
		else if (ace->low >= Half)  {
			bit_plus_follow(ace, 1);
			ace->low -= Half;
			ace->high -= Half;
		}
		else if (ace->low >= First_qtr && ace->high<Third_qtr)  {
			ace->fbits += 1;
			ace->low -= First_qtr;
			ace->high -= First_qtr;
		}
		else
			break;
		ace->low = 2 * ace->low;
		ace->high = 2 * ace->high + 1;
	}

	if (acm->adapt)
		update_model(acm, sym);

	return;
}

int
ac_decode_symbol(ac_decoder *acd, ac_model *acm)
{
	long range;
	int cum;
	int sym;

	range = (long)(acd->high - acd->low) + 1;
	cum = (((long)(acd->value - acd->low) + 1)*acm->cfreq[0] - 1) / range;

	for (sym = 0; acm->cfreq[sym + 1]>cum; sym++)
		/* do nothing */;

	check(sym<0 || sym >= acm->nsym, "symbol out of range");

	acd->high = acd->low + (range*acm->cfreq[sym]) / acm->cfreq[0] - 1;
	acd->low = acd->low + (range*acm->cfreq[sym + 1]) / acm->cfreq[0];

	for (;;)  {
		if (acd->high<Half)  {
			/* do nothing */
		}
		else if (acd->low >= Half)  {
			acd->value -= Half;
			acd->low -= Half;
			acd->high -= Half;
		}
		else if (acd->low >= First_qtr && acd->high<Third_qtr)  {
			acd->value -= First_qtr;
			acd->low -= First_qtr;
			acd->high -= First_qtr;
		}
		else
			break;
		acd->low = 2 * acd->low;
		acd->high = 2 * acd->high + 1;
		acd->value = 2 * acd->value + input_bit(acd);
	}

	if (acm->adapt)
		update_model(acm, sym);

	return sym;
}
//******************************ºâ³N½s½X***************************************

/****************************************SVD°Æµ{¦¡¶}©l***************************************************/
void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr, "Numerical Recipes run-time error...\n");
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	getchar();
	exit(1);
}

float *vectorsvd(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v = (float *)malloc((size_t)((nh - nl + 1 + NR_END)*sizeof(float)));
	if (!v) nrerror("allocation failure in vector()");
	return v - nl + NR_END;
}

void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
	free((FREE_ARG)(v + nl - NR_END));
}

float pythag(float a, float b)
{
	//computes (a^2+b^2)^1/2 without destructive underflow or overflow
	float absa, absb;
	absa = fabs(a);
	absb = fabs(b);
	if (absa > absb)
		return absa*sqrt(1.0 + SQR(absb / absa));

	else
		return (absb == 0.0 ? 0.0 : absb*sqrt(1.0 + SQR(absa / absb)));
}
void svbksb(float **u, float w[], float **v, int m, int n, float b[], float x[])
{
	int jj, j, i;
	float s, *tmp;
	tmp = vectorsvd(1, n);
	for (j = 1; j <= n; j++)
	{
		s = 0.0;//calculate U transpose multiply B
		if (w[j])
		{      // Nonzero result only if wj is nonzero.  
			for (i = 1; i <= m; i++)
				s += u[i][j] * b[i];
			s /= w[j];        //That is the divide by wj.
		}
		tmp[j] = s;
	}
	for (j = 1; j <= n; j++)
	{                  //Matrix multiply by V to get answer.
		s = 0.0;
		for (jj = 1; jj <= n; jj++)
			s += v[j][jj] * tmp[jj];
		x[j] = s;
	}
	free_vector(tmp, 1, n);
}
void svdcmp(float **a, int m, int n, float w[], float **v)
{
	float pythag(float a, float b);
	int flag, i, its, j, jj, k, l, nm;
	float anorm, c, f, g, h, s, scale, x, y, z, *rv1;

	rv1 = vectorsvd(1, n);
	g = scale = anorm = 0.0;
	for (i = 1; i <= n; i++)
	{
		l = i + 1;
		rv1[i] = scale*g;
		g = s = scale = 0.0;
		if (i <= m) {
			for (k = i; k <= m; k++) scale += fabs(a[k][i]);
			if (scale) {
				for (k = i; k <= m; k++) {
					a[k][i] /= scale;
					s += a[k][i] * a[k][i];
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f*g - s;
				a[i][i] = f - g;
				for (j = l; j <= n; j++) {
					for (s = 0.0, k = i; k <= m; k++) s += a[k][i] * a[k][j];
					f = s / h;
					for (k = i; k <= m; k++)
						a[k][j] += f*a[k][i];
				}
				for (k = i; k <= m; k++)
					a[k][i] *= scale;
			}
		}
		w[i] = scale *g;
		g = s = scale = 0.0;
		if (i <= m && i != n) {
			for (k = l; k <= n; k++) scale += fabs(a[i][k]);
			if (scale) {

				for (k = l; k <= n; k++) {
					a[i][k] /= scale;
					s += a[i][k] * a[i][k];
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f*g - s;
				a[i][l] = f - g;
				for (k = l; k <= n; k++)
					rv1[k] = a[i][k] / h;
				for (j = l; j <= m; j++)
				{
					for (s = 0.0, k = l; k <= n; k++)
						s += a[j][k] * a[i][k];
					for (k = l; k <= n; k++)
						a[j][k] += s*rv1[k];
				}
				for (k = l; k <= n; k++) a[i][k] *= scale;
			}
		}
		anorm = FMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}
	for (i = n; i >= 1; i--)
	{    // Accumulation of right-hand transformations.

		if (i < n) {
			if (g) {
				for (j = l; j <= n; j++) //double division to avoid possible underflow.   	  .
					v[j][i] = (a[i][j] / a[i][l]) / g;
				for (j = l; j <= n; j++)
				{
					for (s = 0.0, k = l; k <= n; k++)
						s += a[i][k] * v[k][j];
					for (k = l; k <= n; k++)
						v[k][j] += s*v[k][i];
				}
			}
			for (j = l; j <= n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}
	for (i = IMIN(m, n); i >= 1; i--)
	{ //accumulation of left-hand transformations. 	  	 		
		l = i + 1;
		g = w[i];
		for (j = l; j <= n; j++)
			a[i][j] = 0.0;
		if (g) {
			g = 1.0 / g;
			for (j = l; j <= n; j++) {
				for (s = 0.0, k = l; k <= m; k++) s += a[k][i] * a[k][j];
				f = (s / a[i][i])*g;
				for (k = i; k <= m; k++) a[k][j] += f*a[k][i];
			}
			for (j = i; j <= m; j++) a[j][i] *= g;
		}
		else for (j = i; j <= m; j++) a[j][i] = 0.0;
		++a[i][i];
	}
	for (k = n; k >= 1; k--)
	{                  //Diagonalization of the bidiagonal form;Loop over
		for (its = 1; its <= 30; its++) {    //singular values, and over alloowed iterations. 	  	

			flag = 1;
			for (l = k; l >= 1; l--) { //Test for splitting.
				nm = l - 1;             //Note that rv1[1] is always zero.  	

				if ((float)(fabs(rv1[l]) + anorm) == anorm) {
					flag = 0;
					break;
				}
				if ((float)(fabs(w[nm]) + anorm) == anorm) break;
			}
			if (flag)
			{
				c = 0.0;                        //Cancellation of rv1[1], if l>1.
				s = 1.0;
				for (i = l; i <= k; i++)
				{
					f = s*rv1[i];
					rv1[i] = c*rv1[i];
					if ((float)(fabs(f) + anorm) == anorm)
						break;
					g = w[i];
					h = pythag(f, g);
					w[i] = h;
					h = 1.0 / h;
					c = g*h;
					s = -f*h;
					for (j = 1; j <= m; j++)
					{
						y = a[j][nm];
						z = a[j][i];
						a[j][nm] = y*c + z*s;
						a[j][i] = z*c - y*s;
					}
				}
			}
			z = w[k];
			if (l == k)
			{                      //convergence.
				if (z < 0.0)
				{                //Singular value is made nonnegtive.
					w[k] = -z;
					for (j = 1; j <= n; j++)
						v[j][k] = -v[j][k];
				}
				break;
			}
			// if (its == 30);
			// nrerror("no convergence in 30 svdcmp iterations");
			x = w[l];                 //shift from bottom 2 by 2 minor 
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z)*(y + z) + (g - h)*(g + h)) / (2.0*h*y);
			g = pythag(f, 1.0);
			f = ((x - z)*(x + z) + h*((y / (f + SIGN(g, f))) - h)) / x;
			s = 1.0;                                    //next QR transformation
			c = s;
			for (j = l; j <= nm; j++)
			{
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s*g;
				g = c*g;
				z = pythag(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x*c + g*s;
				g = g*c - x*s;
				h = y*s;
				y *= c;
				for (jj = 1; jj <= n; jj++)
				{
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = x*c + z*s;
					v[jj][i] = z*c - x*s;
				}
				z = pythag(f, h);
				w[j] = z;            //Rotation can be arbitrary if z=o.
				if (z) {
					z = 1.0 / z;
					c = f*z;
					s = h*z;
				}
				f = c*g + s*y;
				x = c*y - s*g;
				for (jj = 1; jj <= m; jj++)
				{
					y = a[jj][j];
					z = a[jj][i];
					a[jj][j] = y*c + z*s;
					a[jj][i] = z*c - y*s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	free_vector(rv1, 1, n);

}
/****************************************SVD°Æµ{¦¡¶}©l**************************************************/
int cholesky(float *aa, float *bb){
	int i, j, k;
	int cholesky_nn = predictor_order;
	float sum, p[MAX_nn_size], a[MAX_nn_size][MAX_nn_size], b[MAX_nn_size], x[MAX_nn_size];
	for (i = 0; i<cholesky_nn; i++){
		for (j = 0; j<cholesky_nn; j++){
			a[i][j] = *(aa + i*cholesky_nn + j);
		}
		b[i] = *(bb + i);
	}
	for (i = 0; i <= cholesky_nn - 1; i++)
	{
		for (j = i; j <= cholesky_nn - 1; j++)
		{
			for (sum = a[i][j], k = i - 1; k >= 0; k--)
				sum -= a[i][k] * a[j][k];
			if (i == j)
			{
				if (sum <= 0.0) //a, with rounding errors, is not positive definite.
				{
					//printf("choldc failed\n");
					return -1;
					exit;
				}
				p[i] = sqrt(sum);
			}
			else a[j][i] = sum / p[i];
		}
	}

	for (i = 0; i <= cholesky_nn - 1; i++)
	{
		//Solve L ¡P y = b, storing y in x.
		for (sum = b[i], k = i - 1; k >= 0; k--) sum -= a[i][k] * x[k];
		x[i] = sum / p[i];
	}
	for (i = cholesky_nn - 1; i >= 0; i--)
	{
		//Solve LT ¡P x = y.
		for (sum = x[i], k = i + 1; k <= cholesky_nn - 1; k++)
			sum -= a[k][i] * x[k];
		x[i] = sum / p[i];
	}
	for (i = 0; i <= cholesky_nn - 1; i++){
		a16[i] = x[i]; //updata 
	}
	//printf("choldc sucessful\n");
	return 0;
}
int svd(float *aa, float *bb){
	int i, j, q;
	int N = predictor_order + 1;

	float w[MAX_nn_size], **v, x[MAX_nn_size], wmax, wmin, **a;
	float test[MAX_nn_size][MAX_nn_size] = { { 0 } }, b[MAX_nn_size] = { 0 };


	a = (float**)calloc(N, sizeof(float*));
	for (q = 0; q<N; q++)
		a[q] = (float*)calloc(N, sizeof(float));

	v = (float**)calloc(N, sizeof(float*));
	for (q = 0; q<N; q++)
		v[q] = (float*)calloc(N, sizeof(float));

	for (i = 1; i<N; i++){
		for (j = 1; j<N; j++){
			test[i][j] = *(aa + (i - 1)*(N - 1) + (j - 1));
		}
		b[i] = *(bb + i - 1);
	}
	for (i = 0; i<N; i++)
		for (j = 0; j<N; j++)
			a[i][j] = test[i][j];
	svdcmp(a, N - 1, N - 1, w, v);

	wmax = 0.0;
	for (j = 1; j<N; j++)
		if (w[j]> wmax)
			wmax = w[j];
	wmin = wmax*pow(10.0, -6);
	for (j = 1; j<N; j++)
		if (w[j] < wmin)
			w[j] = 0.0;
	svbksb(a, w, v, N - 1, N - 1, b, x);
	for (i = 1; i<N; i++)
		a16[i - 1] = x[i];

	for (q = 0; q<N; q++)
		free(a[q]);
	free(a);
	for (q = 0; q<N; q++)
		free(v[q]);
	free(v);
	return 0;
}
/****************************************SVD°Æµ{¦¡¶}©l**************************************************/

/****************************************TranArea******************************************************/
//«Ø¸m³Ì¤p¥­¤èªk
void normalEquation(float *P, float *y, int Tsize){
	float *C, *B;
	C = (float *)malloc(sizeof(float)*(predictor_order));								//«ØºcNormal Equation Bx=C
	B = (float *)malloc(sizeof(float)*(predictor_order*predictor_order));				//«ØºcNormal Equation Bx=C
	memset(C, 0.0, sizeof(float)*(predictor_order));
	memset(B, 0.0, sizeof(float)*(predictor_order*predictor_order));
	for (int i = 0; i<predictor_order; i++){
		for (int j = 0; j<predictor_order; j++){
			for (int k = 0; k<Tsize; k++){
				*(B + i * predictor_order + j) += (*(P + k * predictor_order + i))*(*(P + k * predictor_order + j)); //Pt x P
			}
		}
	}
	for (int i = 0; i<predictor_order; i++){
		for (int j = 0; j<Tsize; j++){
			*(C + i) += (*(P + j * predictor_order + i))*(*(y + j));//Pt x y
		}
	}
	choleskyCount++;
	if (cholesky(B, C) != 0){
		svd(B, C);
		svdCount++;
	}
	free(B);
	free(C);
}
//«Øºc°V½m°Ï¶¡
void tranAreaRegular4(int a, int *image){

	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width);//x
	int m = floor(a / (Width));//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	/*
	printf("*****************************\n");
	printf("x:%d\ty:%d\n",n,m);
	printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	printf("TSize:%d\n",TSize);
	printf("*****************************\n");
	getchar();
	*/
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a 
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(c + index) = *(image + (x + y * Width));
		index++;
	}

	normalEquation(P, c, TSize);
	free(P);
	free(c);
}
void tranAreaRegular6(int a, int *image){

	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width);//x
	int m = floor(a / (Width));//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	//printf("x:%d\ty:%d\n",n,m);
	//printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	//printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	//printf("TSize:%d\n",TSize);
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
			*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
		*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
		*(c + index) = *(image + (x + y * Width));
		index++;
	}

	normalEquation(P, c, TSize);
	free(P);
	free(c);
}
void tranAreaRegular8(int a, int *image){

	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width);//x
	int m = floor(a / (Width));//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	//printf("x:%d\ty:%d\n",n,m);
	//printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	//printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	//printf("TSize:%d\n",TSize);
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
			*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
			*(P + index*predictor_order + 6) = *(image + (x + y * Width) - Width - 2);//nww
			*(P + index*predictor_order + 7) = *(image + (x + y * Width) - 2 * Width - 1);//nnw
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
		*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
		*(P + index*predictor_order + 6) = *(image + (x + y * Width) - Width - 2);//nww
		*(P + index*predictor_order + 7) = *(image + (x + y * Width) - 2 * Width - 1);//nnw
		*(c + index) = *(image + (x + y * Width));
		index++;
	}

	normalEquation(P, c, TSize);
	free(P);
	free(c);
}
void tranAreaRegular10(int a, int *image){

	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width);//x
	int m = floor(a / (Width));//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 3;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 3;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	//printf("x:%d\ty:%d\n",n,m);
	//printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	//printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	//printf("TSize:%d\n",TSize);
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
			*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
			*(P + index*predictor_order + 6) = *(image + (x + y * Width) - Width - 2);//nww
			*(P + index*predictor_order + 7) = *(image + (x + y * Width) - 2 * Width - 1);//nnw
			*(P + index*predictor_order + 8) = *(image + (x + y * Width) - 2 * Width + 1);//nne
			*(P + index*predictor_order + 9) = *(image + (x + y * Width) - Width + 2);//nee
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
		*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
		*(P + index*predictor_order + 6) = *(image + (x + y * Width) - Width - 2);//nww
		*(P + index*predictor_order + 7) = *(image + (x + y * Width) - 2 * Width - 1);//nnw
		*(P + index*predictor_order + 8) = *(image + (x + y * Width) - 2 * Width + 1);//nne
		*(P + index*predictor_order + 9) = *(image + (x + y * Width) - Width + 2);//nee
		*(c + index) = *(image + (x + y * Width));
		index++;
	}

	normalEquation(P, c, TSize);
	free(P);
	free(c);
}

/****************************************TranArea******************************************************/

int preValue(int x1, int x2, int x3, int x4, int x5, int x6, int x7, int x8, int x9, int x10){
	if (predictor_order == 4){
		return (int)(((a16[0] * x1)//a
			+ (a16[1] * x2)//b
			+ (a16[2] * x3)//c
			+ (a16[3] * x4))); //   ) / (a16[0] + a16[1] + a16[2] + a16[3] )); //4 d
	}
	if (predictor_order == 6){
		return (int)(((a16[0] * x1)//a
			+ (a16[1] * x2)//b
			+ (a16[2] * x3)//c
			+ (a16[3] * x4) //   ) / (a16[0] + a16[1] + a16[2] + a16[3] )); //4 d
			//N=6
			+ (a16[4] * x5) //e
			+ (a16[5] * x6))); // (a16[0] + a16[1] + a16[2] + a16[3] + a16[4] + a16[5]));//f
	}
	if (predictor_order == 8){
		return (int)(((a16[0] * x1)//a
			+ (a16[1] * x2)//b
			+ (a16[2] * x3)//c
			+ (a16[3] * x4) //   ) / (a16[0] + a16[1] + a16[2] + a16[3] )); //4 d
			//N=6
			+ (a16[4] * x5) //e
			+ (a16[5] * x6) // (a16[0] + a16[1] + a16[2] + a16[3] + a16[4] + a16[5]));//f
			//N=8
			+ (a16[6] * x7) //7
			+ (a16[7] * x8)));
	}
	if (predictor_order == 10){
		return (int)(((a16[0] * x1)//a
			+ (a16[1] * x2)//b
			+ (a16[2] * x3)//c
			+ (a16[3] * x4) //   ) / (a16[0] + a16[1] + a16[2] + a16[3] )); //4 d
			//N=6
			+ (a16[4] * x5) //e
			+ (a16[5] * x6) // (a16[0] + a16[1] + a16[2] + a16[3] + a16[4] + a16[5]));//f
			//N=8
			+ (a16[6] * x7) //7
			+ (a16[7] * x8)
			//N=10
			+ (a16[8] * x9) //7
			+ (a16[9] * x10)));
	}
}
//OpenCV Slover
void cvSVD(int a, int *image){

	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width)+1;//x
	int m = a / (Width)+1;//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	//printf("x:%d\ty:%d\n",n,m);
	//printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	//printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	//printf("TSize:%d\n",TSize);
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
			*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
		*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
		*(c + index) = *(image + (x + y * Width));
		index++;
	}

	CvMat *Matrix1 = cvCreateMat(TSize, predictor_order, CV_32FC1);
	CvMat *Matrix2 = cvCreateMat(TSize, 1, CV_32FC1);
	CvMat *SolveSet = cvCreateMat(predictor_order, 1, CV_32FC1);


	//printf("*******************************************************************************\n");

	cvSetData(Matrix1, P, Matrix1->step);
	cvSetData(Matrix2, c, Matrix2->step);
	cvSolve(Matrix1, Matrix2, SolveSet, CV_SVD);

	//PrintMatrix(Matrix1,Matrix1->rows,Matrix1->cols);
	//PrintMatrix(Matrix2,Matrix2->rows,Matrix2->cols);
	//PrintMatrix(SolveSet,SolveSet->rows,SolveSet->cols);
	//printf("%d",Matrix1->rows);
	for (int l = 0; l<predictor_order; l++){
		a16[l] = cvGet1D(SolveSet, l).val[0];;
	}
	//printf("a16: %f,%f,%f,%f,%f,%f\n",a16[0],a16[1],a16[2],a16[3],a16[4],a16[5]);
	free(P);
	free(c);
}

int EdgeDetect(int i, unsigned char *image){
	float average, variance, std_deviation, sum = 0, sum1 = 0;
	int x[4];
	int kh[2], kl[2];
	float aveh, varh;
	float avel, varl;
	x[0] = *(image + i - 1);//x1
	x[1] = *(image + i - Width);//x2
	x[2] = *(image + i - Width - 1);//x3
	x[3] = *(image + i - Width + 1);//x4
	average = (x[0] + x[1] + x[2] + x[3]) / (float)4;//x4
	variance = (pow((x[0] - average), 2)//x1
		+ pow((x[1] - average), 2)//x2
		+ pow((x[2] - average), 2)//x3
		+ pow((x[3] - average), 2)) / (float)4;//x4
	for (i = 0; i<4; i++){
		int h = 0, l = 0;
		if (x[i] - average < 0){
			kh[h++];
		}
		if (x[i] - average > 0){
			kl[l++];
		}
	}
	if (variance >= 100){
		aveh = (kh[0] + kh[1]) / (float)2;
		varh = (pow((kh[0] - aveh), 2)
			+ pow((kh[1] - aveh), 2)) / (float)2;

		avel = (kl[0] + kl[1]) / (float)2;
		varl = (pow((kl[0] - avel), 2)
			+ pow((kl[1] - avel), 2)) / (float)2;
		if (variance >= (10 * (varh + varl))){
			return 1;
		}
	}
	return 0;
}
//Medium Edge Dector
int MED(){
	if (nw >= MAX(w, n)){
		return MIN(w, n);
	}
	else if (nw <= MIN(w, n)){
		return MAX(w, n);
	}
	else{
		return n + (w - nw);
	}
}

int PredictRange(int xp){
	if (xp > 255)
		return xp = 255;
	else if (xp < 0)
		return xp = 0;
	else
		return xp;
}

unsigned char ErrorStrength(int delta){
	
	//4 Section

	if (delta < q[1])
		return (delta < q[0]) ? 0 : 1;
	else
		return (delta < q[2]) ? 2 : 3;

}

int readArithmeticCoding(int quantizerEnergy){
	int Decode = 0;
	switch (quantizerEnergy){
	case 0:
		Decode = ac_decode_symbol(&acd, &acm1);
		break;
	case 1:
		Decode = ac_decode_symbol(&acd, &acm2);
		break;
	case 2:
		Decode = ac_decode_symbol(&acd, &acm3);
		break;
	case 3:
		Decode = ac_decode_symbol(&acd, &acm4);
		break;
	case 4:
		Decode = ac_decode_symbol(&acd, &acm5);
		break;
	case 5:
		Decode = ac_decode_symbol(&acd, &acm6);
		break;
	case 6:
		Decode = ac_decode_symbol(&acd, &acm7);
		break;
	case 7:
		Decode = ac_decode_symbol(&acd, &acm8);
		break;
	default:
		break;
	}
	return Decode;
}

int AdatpDecoder(int quantizerEnergy){
	int DecodeValue = 0;
	//quantizerEnergy = 7;
	int escape = N[quantizerEnergy] - 1;
	//printf("%d\n", quantizerEnergy);
	DecodeValue = readArithmeticCoding(quantizerEnergy);

	if (quantizerEnergy < 7)
		if (DecodeValue == escape)
			DecodeValue += AdatpDecoder(quantizerEnergy++);

	return DecodeValue;
	
}

int quantize(double a)
{
	int NC = 7;
	int i;

	if (a<q[0]) return(0);
	else if (a >= q[NC - 1]) return(NC);
	else
		for (i = 1; i<NC; i++)
			if (a >= q[i - 1] && a<q[i])
				return(i);
}
void RegularMode(int* ReSourceImage, int* ErrorImage){
	printf("Decoding....\n");
	bool pureMED = false;
	//bool AllLSR = 1;
	bool SettingN = 1;
	bool isLSR = false;
	bool isMED = false;
	int th = 3;
	int ep = 0;
	int error = 0;
	int DecodeValue = 0;
	for (int i = 1; i < (Width)*(Height)*(Pixel_Byte); i++){

		if (predictor_order != PredictorOrderSetting && SettingN){
			printf("Only DO N = %d\n", PredictorOrderSetting);
			break;//do only N
		}
		if (((((i % (Width)) != 0)) && ((i >= (Width)))) && !pureMED){	//²Ä0¦C¥H¤Î²Ä0¦æ¥H¥~
			w = *(ReSourceImage + i - 1);//1
			n = *(ReSourceImage + i - Width);//2
			nw = *(ReSourceImage + i - Width - 1);//3
			ne = *(ReSourceImage + i - Width + 1);//4
			if (predictor_order == 4){
				if (((i % (Width)) < 2) || (i < (2 * Width)) || ((i % (Width)) >(Width - 2))){//²Ä1¦C,²Ä1¦æ,Width-1¦æ   MED
					//|| (abs(nw - w) < th && abs(nw - n) < th && abs(n - ne) < th)){
					//if (((i % (Width)) == 1) || (i <= (2 * Width)) || ((i % (Width)) >(Width - 2))){//²Ä1¦C,²Ä1¦æ,Width-1¦æ MED

					predicted = MED();
					isLSR = false;
					isMED = true;
				}
				else{
					if (AllLSR || (abs(n - w)) >= 12 || *(ErrorImage + i - 1) >= 8)
						tranAreaRegular4(i, ReSourceImage);
					predicted = preValue(w, n, nw, ne, 0, 0, 0, 0, 0, 0);
					isLSR = true;
					isMED = false;
				}
			}
			if (predictor_order == 6){
				if (((i % (Width)) <= 2) || (i < (3 * Width)) || ((i % (Width)) >(Width - 2))){//²Ä1¦C,²Ä1¦æ,²Ä2¦C,²Ä2¦æ,Width-1¦æ   MED
					//|| (abs(nw - w) < th && abs(nw - n) < th && abs(n - ne) < th)){
					//if (((i % (Width)) == 1) || (i <= (2 * Width)) || ((i % (Width)) >(Width - 2))){//²Ä1¦C,²Ä1¦æ,Width-1¦æ MED

					predicted = MED();
					isLSR = false;
					isMED = true;
				}
				else{
					ww = *(ReSourceImage + i - 2);//5
					nn = *(ReSourceImage + i - 2 * Width);//6
					if (AllLSR || (abs(n - w)) >= 12 || *(ErrorImage + i - 1) >= 8)
						tranAreaRegular6(i, ReSourceImage);
					predicted = preValue(w, n, nw, ne, ww, nn, 0, 0, 0, 0);
					isLSR = true;
					isMED = false;
				}
			}
			if (predictor_order == 8){
				if (((i % (Width)) <= 2) || (i < (3 * Width)) || ((i % (Width)) >(Width - 2))){//²Ä1¦C,²Ä1¦æ,²Ä2¦C,²Ä2¦æ,Width-1¦æ   MED
					//|| (abs(nw - w) < th && abs(nw - n) < th && abs(n - ne) < th)){
					//if (((i % (Width)) == 1) || (i <= (2 * Width)) || ((i % (Width)) >(Width - 2))){//²Ä1¦C,²Ä1¦æ,Width-1¦æ MED

					predicted = MED();
					isLSR = false;
					isMED = true;
				}
				else{
					ww = *(ReSourceImage + i - 2);//5
					nn = *(ReSourceImage + i - 2 * Width);//6
					int nww = *(ReSourceImage + i - Width - 2);//7
					int nnw = *(ReSourceImage + i - 2 * Width - 1);//8
					if (AllLSR || (abs(n - w)) >= 12	|| *(ErrorImage + i - 1) >= 8)
						tranAreaRegular8(i, ReSourceImage);
					predicted = preValue(w, n, nw, ne, ww, nn, nww, nnw, 0, 0);
					isLSR = true;
					isMED = false;
				}
			}
			if (predictor_order == 10){
				if (((i % (Width)) <= 2) || (i < (3 * Width)) || ((i % (Width)) >(Width - 3))){//²Ä1¦C,²Ä1¦æ,²Ä2¦C,²Ä2¦æ,Width-1¦æ   MED
					//|| (abs(nw - w) < th && abs(nw - n) < th && abs(n - ne) < th)){
					//if (((i % (Width)) == 1) || (i <= (2 * Width)) || ((i % (Width)) >(Width - 2))){//²Ä1¦C,²Ä1¦æ,Width-1¦æ MED

					predicted = MED();
					isLSR = false;
					isMED = true;
				}
				else{
					ww = *(ReSourceImage + i - 2);//5
					nn = *(ReSourceImage + i - 2 * Width);//6
					int nww = *(ReSourceImage + i - Width - 2);//7
					int nnw = *(ReSourceImage + i - 2 * Width - 1);//8
					int nne = *(ReSourceImage + i - 2 * Width + 1);//9
					int nee = *(ReSourceImage + i - Width + 2);//10
					if (AllLSR || (abs(n - w)) >= 12 || *(ErrorImage + i - 1) >= 8)
						tranAreaRegular10(i, ReSourceImage);
					predicted = preValue(w, n, nw, ne, ww, nn, nww, nnw, nne, nee);
					isLSR = true;
					isMED = false;
				}
			}
		}
		else {
			if (i == 0){//0,0
				predicted = 0;
				isLSR = false;
				isMED = false;
			}
			else if (i%Width == 0){//0,n ²Ä0¦æ
				predicted = *(ReSourceImage + i - Width);
				isLSR = false;
				isMED = false;
			}
			else if (i < Width){//n,0 ²Ä0¦C
				predicted = *(ReSourceImage + i - 1);
				isLSR = false;
				isMED = false;
			}
			else if (pureMED){
				w = *(ReSourceImage + i - 1);//1
				n = *(ReSourceImage + i - Width);//2
				nw = *(ReSourceImage + i - Width - 1);//3
				predicted = MED();
				isLSR = false;
				isMED = true;
			}
		}
		predicted = PredictRange(predicted);

		unsigned char quantizerTexture = 0;
		unsigned char coffiTexture = 0;
		unsigned char contextIndex = 0;
		unsigned char quantizerEnergy = 7;
		double delta = 0;
		float RoundTemp = 0;
		if (contextLSR && i > (2 * Width) && (i % Width) > 1 && (i % Width) < (Width - 1)){
			//if (((i % (Width)) > 2) && (i > (3 * Width)) && ((i % (Width)) < (Width - 2))){

			//tranAreaRegular6(i, ErrorImage);

			int En = *(ErrorImage + i - Width);
			int Enn = *(ErrorImage + i - 2 * Width);
			int Enw = *(ErrorImage + i - Width - 1);
			int Ew = *(ErrorImage + i - 1);
			int Eww = *(ErrorImage + i - 2);
			int Ene = *(ErrorImage + i - Width + 1);
			int Enne = *(ErrorImage + i - 2 * Width + 1);
			int Ewwn = *(ErrorImage + i - 2 - Width);
			//delta = preValue(Ew, En, Enw, Ene, Eww, Enn, 0, 0, 0, 0);//EDP
			//delta = PredictRange(delta);
			double powd;
			powd = pow(abs(En), 2) + pow(abs(Ew), 2) + pow(abs(Enw), 2) + pow(abs(Ene), 2) + pow(abs(Enn), 2) + pow(abs(Eww), 2) + pow(abs(Enne), 2) + pow(abs(Ewwn), 2);
			delta = pow((powd / 8), 0.5) * 10;
			//delta = 0.0034 * (abs(ne - n) + abs(n - nw) + abs(nw - w)) + 0.048 * abs(Ew);
			quantizerEnergy = quantize(delta);
		}

		//quantizerEnergy = 3;
		DecodeValue = AdatpDecoder(quantizerEnergy);
		//DecodeValue = ac_decode_symbol(&acd, &acm4);

		
		//if (ep < 0)
		//DecodeValue -= 128;
		DecodeValue = (DecodeValue % 2 == 0) ? (DecodeValue >> 1) : -(DecodeValue + 1) >> 1;

		//*
		if (DecodeValue < -(predicted + ep))
			DecodeValue += 256;
		if (DecodeValue > (255 - (predicted + ep)))
			DecodeValue -= 256;
		//*/
		//error = DecodeValue;

		//x = *(ReSourceImage + i) = (DecodeValue + (predicted + ep));
		x = *(ReSourceImage + i) = (DecodeValue + predicted);
		*(ErrorImage + i) = x - predicted;

		//ep = 0;
		//quantizerEnergy = 7;
		/*
		if (contextLSR && i > 2 * Width && i % Width > 1 && i % Width < Width - 1){
			S[contextIndex] += (x - predicted);
			context_count[contextIndex] ++;	//Context­pºâ
			if (context_count[contextIndex] == 128) {
				context_count[contextIndex] >>= 1;
				S[contextIndex] >>= 1;
			}
		}
		*/
	}
}

int main(int argc, char *argv[])
{
#if TEST_ONE
	argc = 5;
	argv[1] = "Lennagrey";
	argv[2] = "6";
#endif
	if (argc == 5) {
		printf("-------------------DECODING START---------------------------\n");
		FILEName = argv[1];
		printf("%s\n", FILEName);
		char* InBitName = OutFileName(FILE_PATH, FILEName, argv[2], "Encode");//¿é¤JÀÉ¦W

		char* OutImageName = OutFileName(FILE_OUT_PATH, FILEName, argv[2], "Decode.bmp");//¿é¥X¼v¹³ÀÉ¦W

		char UWidth;//Width Upper 2 Bit
		char LWidth;//Width Lower 2 Bit
		char UHeight ;//Height Upper 2 Bit
		char LHeight ;//Height Lower 2 Bit
		int *ReSourceImage, *ErrorImage;
		int FirstPixel;

		FILE *fp;
		if ((fp = fopen(OutFileName(FILE_PATH, FILEName, argv[2], "Encode"), "rb")) == 0){
			printf("Read File Header Fail\n");
			system("pause");
			return 0;
		}
		else{
			//fseek(fp, 0, SEEK_SET);
			UWidth = getc(fp);//Width Upper 2 Bit
			LWidth = getc(fp);//Width Lower 2 Bit
			UHeight = getc(fp);//Height Upper 2 Bit
			LHeight = getc(fp);//Height Lower 2 Bit

			predictor_order = getc(fp);//Predict Size
			FirstPixel = getc(fp);
			fclose(fp);
			printf("Read File Header...OK!\n");
		}

		Width = (UWidth << 8) | LWidth;
		Height = (UHeight << 8) | LHeight;
		//Width = 512;
		//Height = 512;
		//predictor_order = 6;
		printf("%d\n", FirstPixel);
		Mat image(Height, Width, CV_8U);

		ReSourceImage = (int *)malloc(sizeof(int)*((Width)*(Height)*(Pixel_Byte)));  //¥Î¥HÀx¦s­ì©l¼v¹³ªºªÅ¶¡
		ErrorImage = (int *)malloc(sizeof(int)*((Width)*(Height)*(Pixel_Byte)));  //¥Î¥HÀx¦s»~®t¼v¹³ªºªÅ¶¡ 

		ac_decoder_init(&acd, OutFileName(FILE_PATH, FILEName, argv[2], "Encode"));//ºâ¼Æ½s½XInit
		ac_model_init(&acm1, N[0], NULL, ADAPT);
		ac_model_init(&acm2, N[1], NULL, ADAPT);
		ac_model_init(&acm3, N[2], NULL, ADAPT);
		ac_model_init(&acm4, N[3], NULL, ADAPT);
		ac_model_init(&acm5, N[4], NULL, ADAPT);
		ac_model_init(&acm6, N[5], NULL, ADAPT);
		ac_model_init(&acm7, N[6], NULL, ADAPT);
		ac_model_init(&acm8, N[7], NULL, ADAPT);


		*(ReSourceImage) = FirstPixel;
		*(ErrorImage) = *(ReSourceImage);

		compress_start = clock();	  //§ì¨úµ{¦¡¶}©l°õ¦æ¤§¨t²Î®É¶¡

		compress_start2 = clock();
		RegularMode(ReSourceImage, ErrorImage);

		compress_end = clock();

#if ADAPTIVE_ARITHMETIC_CODING
		ac_decoder_done(&acd);
		ac_model_done(&acm1);
		ac_model_done(&acm2);
		ac_model_done(&acm3);
		ac_model_done(&acm4);
		ac_model_done(&acm5);
		ac_model_done(&acm6);
		ac_model_done(&acm7);
		ac_model_done(&acm8);
#endif

		WriteImage(ReSourceImage, image.data);
		imwrite(OutImageName, image);//Gray Image

		free(ReSourceImage);
		free(ErrorImage);

		printf("-------------------DECODING END---------------------------\n");
	}
	else if (argc > 6) {
		printf("Too many arguments supplied.\nFILEName\tWidth\tHeight\tpredictor_order\t\n");
	}
	else {
		printf("One argument expected.\nFILEName\tWidth\tHeight\tpredictor_order\t\n");
	}
	//system("pause");
	return 0;
}
//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

// CREATE UNIGRAM TABLE TO STORE DISTRO BASED ON VOCAB SIZE
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  // ch: CHARACTER HOLDER
  // a: USED TO COUNT LEN OF WORD
  int a = 0, ch;
  // WHILE THE INPUT FILE ISN'T AT EOF MARK
  while (!feof(fin)) {
    // GET NEXT CHAR
    ch = fgetc(fin);
    // IF CARRIAGE RETURN CONTINUE LOOP
    if (ch == 13) continue;

    // CHECK IF CHARACTER IS SPACE, TAB, OR NEW LINE
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      // IF WE HAVE A WORD CHECK THAT WE AREN'T ADDING
      // A NEW LINE CHAR TO IT
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      // IF EMPTY LINE SET WORD TO SPACE TOKEN STR
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    // ADD CHAR TO WORD
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  // END WORD
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  // a: USED TO ITER THRU CURR WORD
  // hash: USED TO HOLD HASH VAL
  unsigned long long a, hash = 0;
  // CALC HASH VAL FOR WORD
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  // ASSIGN HASH VAL W/IN LIMIT OF HASH TAB
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  // GET HASH ID FOR WORD
  unsigned int hash = GetWordHash(word);
  // LOOP THRU VOCAB HASH
  while (1) {
    // IF NO VALUE ASSIGNED TO VOCAB HASH, RETURN -1
    if (vocab_hash[hash] == -1) return -1;
    // IF THE WORD IN THE CURRENT LOCATION MATCHES RETURN THE VAL
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    // INCREMENT IN THE CASE THE WORDS DIDN'T MATCH
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  // HOLD WORD FROM FILE
  char word[MAX_STRING];
  ReadWord(word, fin);
  // IF FILE AT END, THEN NO INDEX TO RETURN
  if (feof(fin)) return -1;
  // IF WORD EXISTS GET IDX IN VOCAB
  return SearchVocab(word);
}

// Adds a word to the vocabulary
// ADD WORD TO VOCAB, SET WORD COUNT TO 0
// GET HASH ID FOR WORD, SET VOCAB HASH
// TO THE CURR VOCAB SIZE
int AddWordToVocab(char *word) {
  // hash: HOLDS HASH ID
  // length: LEN OF WORD 
  unsigned int hash, length = strlen(word) + 1;
  // IF WORD TOO LONG, LIMIT LEN
  if (length > MAX_STRING) length = MAX_STRING;
  // ALLOC MEM FOR WORD IN VOCAB LIST
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  // COPY WORD TO VOCAB LIST
  strcpy(vocab[vocab_size].word, word);
  // SET COUNT FOR THE WORD TO 0
  vocab[vocab_size].cn = 0;
  // INCREMENT VOCAB COUNT
  vocab_size++;
  // Reallocate memory if needed
  // NEARLY 1k WORDS, ADD MEM
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  // GET HASH ID FOR WORD
  hash = GetWordHash(word);
  // CHECK IF HASH ID IS ALREADY USED, IF IT IS GO TO NEXT AVAIL
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  // ASSIGN VOCAB HASH THE CURR VOCAB IDX
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
// SORTING FXN TO COMPARE WORD COUNT OF 2 WORDS
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// SORT VOCAB BY WORD FREQ, REMOVE WORDS W/ LOW COUNTS FROM VOCAB AND
// REMOVE FROM HASH, SET TRAINING WORD COUNT, REDUCE MEM USAGE FOR VOCAB
// SETUP BINARY HUFFMAN TREE
void SortVocab() {
  // a: USED TO IDX THRU VOCAB
  // size: USED TO HOLD VOCAB SIZE LOCALLY
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);

  // FILL VOCAB HASH TABLE W/ -1
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  // SET SIZE TO VOCAB SIZE
  size = vocab_size;
  // SET COUNT OF WORDS IN TRAINING FILE TO 0
  train_words = 0;
  // LOOP THRU VOCAB
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      // CHECK IF HASH ID IS ALREADY USED, IF IT IS GO TO NEXT AVAIL
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      // ASSIGN VOCAB HASH THE CURR VOCAB IDX
      vocab_hash[hash] = a;
      // INCREMENT TRAINING WORD COUNT BY THE COUNT OF THIS WORD
      train_words += vocab[a].cn;
    }
  }
  // WE CAN DECREASE OUR MEMORY USAGE NOW
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the Huffman binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
// REMOVES LOW USAGE WORDS, REASSIGNS HASHES BASED ON REDUCED SET
// INCREASES THE MIN SIZE EACH CALL
void ReduceVocab() {
  // a: USED TO ITER THRU VOCAB
  // b: USED TO KEEP REDUCED VOCAB COUNT
  int a, b = 0;
  unsigned int hash;

  // ITER THRU VOCAB, IF USAGE COUNT IS HIGHER THAN CURR MIN
  // KEEP THE WORD, ELSE REMOVE THE WORD
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  // REDUCE VOCAB
  vocab_size = b;
  // FILL VOCAB HASH TABLE W/ -1
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    // CURR VOCAB HASH USED, INCREMENT TO NEXT OPEN
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  // INCREASE THE MIN FOR NEXT TIME
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

// SETUP VOCAB FROM TRAINING FILE SUPPLIED
// ADD TRAINING FILE TO VOCAB AND VOCAB HASH
// REMOVE LOW USAGE WORDS 
void LearnVocabFromTrainFile() {
  // word: USED TO HOLD WORDS FROM FILE
  char word[MAX_STRING];
  // DECLARE FILE
  FILE *fin;
  // a: USED TO INCREMENT THRU VOCAB HASH
  // i: USED TO HOLD IDX OF VOCAB
  long long a, i;

  // FILL VOCAB HASH TABLE W/ -1
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  // OPEN UP TRAINING DATA IF IT EXISTS
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  // CLEAR VOCAB COUNT
  vocab_size = 0;

  // ADD THE SPACE TOKEN TO VOCAB TO START 
  AddWordToVocab((char *)"</s>");

  // LOOP THRU WORDS IN FILE ASSIGN TO VOCAB AND VOCAB HASH
  // IF VOCAB TOO LARGE REMOVE LOW USAGE WORDS
  while (1) {
    // OBTAIN SINGLE WORD FROM FILE
    ReadWord(word, fin);
    // IF FILE NOW EMPTY, STOP LOOP
    if (feof(fin)) break;
    // INCREMENT TRAINING WORD COUNT
    train_words++;

    // IF USER IS DEBUGGING REPORT WORD COUNT
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }

    // GET THE IDX OF THE VOCAB IF IT EXISTS 
    i = SearchVocab(word);
    // CHECK IF VOCAB IN HASH, IF NOT ADD IT
    if (i == -1) {
      // a: VOCAB SIZE
      // ADD WORD TO VOCAB, ADD VOCAB IDX TO
      // THE VOCAB HASH
      a = AddWordToVocab(word);
      // INCREMENT VOCAB COUNT
      vocab[a].cn = 1;
    } else vocab[i].cn++;

    // IF MOST OF OUR HASH IS USED, CLEAR LOW USAGE WORDS
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }

  // REMOVE LOW USAGE VOCAB, SETUP B-TREE
  SortVocab();
  // CHECK IF USER IS DEBUGGING AND REPORT STATS
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  // JUMP TO END OF FILE TO REPORT FILE SIZE
  file_size = ftell(fin);
  fclose(fin);
}

// INCREMENT THRU VOCAB AND STORE IN REQUESTED FILE
void SaveVocab() {
  // USED TO INCR THRU VOCAB
  long long i;
  // OPEN FILE FOR SAVING VOCAB
  FILE *fo = fopen(save_vocab_file, "wb");
  // STORE THE WORD AND THE WORD COUNT FOR THE ENTIRE VOCAB
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

// TRY TO READ PRE-LEARNED VOCAB, ADD WORDS TO VOCAB AND VOCAB HASH
// REMOVE LOW COUNT WORDS,COUNT THE TRAINING WORDS THAT WILL BE USED
void ReadVocab() {
  // a: USED TO SET VOCAB HASH TO -1, HOLDS VOCAB IDX
  // i: USED TO COUNT WORDS IN FILE
  long long a, i = 0;
  char c;
  // word: USED TO HOLD WORD FROM FILE
  char word[MAX_STRING];
  // OPEN THE PRE-LEARNED VOCAB FILE, SHOULD BE NON TEXT FORMAT
  FILE *fin = fopen(read_vocab_file, "rb");
  // FAILED TO OPEN FILE, OR FILE EMPTY
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }

  // FILL VOCAB HASH TABLE W/ -1
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  // CLEAR VOCAB COUNT
  vocab_size = 0;

  // LOOP THRU ALL WORDS IN FILE
  // ADD WORDS TO VOCAB AND HASH, SET WORD COUNT. SET VOCAB SIZE
  while (1) {
    // OBTAIN SINGLE WORD FROM FILE
    ReadWord(word, fin);
    // IF FILE NOW EMPTY, STOP LOOP
    if (feof(fin)) break;
    // a: VOCAB SIZE
    // ADD WORD TO VOCAB, ADD VOCAB IDX TO
    // THE VOCAB HASH
    a = AddWordToVocab(word);
    // SET WORD COUNT AND C
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }

  // REMOVE LOW USAGE VOCAB, SETUP B-TREE
  SortVocab();
  // CHECK IF USER IS DEBUGGING AND REPORT STATS
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }

  // OPEN UP TRAINING DATA IF IT EXISTS
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  // JUMP TO END OF TRAINING FILE
  fseek(fin, 0, SEEK_END);
  // GET THE LOCATION OF THE END OF THE FILE TO GET FILE SIZE
  file_size = ftell(fin);
  fclose(fin);
}

// INIT HIDDEN LAYERS OF NETWORK, ASSIGN HUFFMAN CODES TO VOCAB
void InitNet() {
  // a: USED TO ITER THRU VOCAB SIZE TO INIT HIDDEN LAYERS
  // b: USED TO ITER THRU VOCAB SIZE TO INIT HIDDEN LAYERS
  long long a, b;
  unsigned long long next_random = 1;
  // ALLOC MEM FOR HIDDEN LAYER SYN0 ON ALIGNMENT TO THE MIN SIZE REQ
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  // IF IT IS HIERARCHICAL SOFTMAX
  if (hs) {
    // ALLOC MEM FOR HIDDEN LAYER SYN1 ON ALIGNMENT TO THE MIN SIZE REQ
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    // INIT HIDDEN LAYER W/ 0s
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }

  // IF USING NEG SAMPLING
  if (negative>0) {
    // ALLOC MEM FOR HIDDEN LAYER SYN1NEG ON ALIGNMENT TO THE MIN SIZE REQ
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}

    // INIT HIDDEN LAYER W/ 0s
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }

  // INIT HIDDEN LAYER W/ RAND VALS
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }

  // ASSIGN HUFFMAN CODES TO VOCAB
  CreateBinaryTree();
}

// CALC WEIGHTS FOR MODEL 
void *TrainModelThread(void *id) {
  // a: USED TO ITER THRU WINDOW
  // b: USED TO SET WINDOW SIZE
  // d: USED TO SWEEP THRU NEG SAMPLING SIZE
  // cw: HOLDS COUNT OF WORDS FOR WINDOW
  // word: HOLDS WORD IDX
  // last_word: HOLDS THE WORD IDX IN THE SENTENCE
  // sentence_length: HOLDS SIZE OF SENTENCE
  // sentence_position: HOLDS CURRENT LOC IN SENTENCE
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  // word_count: HOLDS OVERALL WORD CONT
  // last_word_count: HOLDS OUR LAST SAVED WORD COUNT FOR REPORTING PURPOSES
  // sen: HOLDS VOCAB IDX VALS FOR WORDS FROM FILE
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  // l1: USED FOR SIGMOID CALCS FOR HIERARCHICAL SOFTMAX
  // l2: USED FOR SIGMOID CALCS FOR HIERARCHICAL SOFTMAX
  // c: USED FOR ITER THRU LOCAL WEIGHTS
  // target: USED TO SET INIT WORD
  // label: USED TO DECIDE BETWEEN ACTUAL WORD AND NEG SAMPLES
  // local_iter: KEEPS TRACK OF TRAINING ITER RUN BY THREADS
  long long l1, l2, c, target, label, local_iter = iter;
  // SET NEXT RANDOM EQUAL TO THREAD ID
  unsigned long long next_random = (long long)id;
  // f: USED AS FREQ FOR SIGMOID CALCS FOR HIERARCHICAL SOFTMAX
  // g: USED AS GRAD FOR WEIGHT UPDATES FOR HIERARCHICAL SOFTMAX
  real f, g;
  // USED FOR STORING TIME
  clock_t now;
  // USED TO HOLD LOCAL LAYER VALS
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  // OPEN TRAINING FILE
  FILE *fi = fopen(train_file, "rb");

  // SEEK TO LOCATION FOR THREAD IN FILE
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  
  // IF THE WORD COUNT IS OVER 10K HIGHER THAN LAST TIME UPDATE ACTUAL 
  // WORD COUNT. REPORT THE STATUS OF MODEL IF USER IS IN DEBUG. AFTER THE
  // 10K ALSO REDUCE THE LR.
  // IF THERE IS NO SENTENCE LEN, ITERATE THRU 1K WORDS OR UNTIL EOF
  // ADDING EACH TO THE SENTENCE. IF IS EOF OR WE HAVE MORE WORDS THAN 
  // TRAINING PER THREAD ALLOWS INCREASE ITER. STOP IF DONE ELSE, RESET 
  // AFTER CREATING SENTENCE TRAIN MODEL FOR ALL WORDS IN SENTENCE
  while (1) {

    // IF THE WORD COUNT IS OVER 10K LARGER THAN LAST TIME UPDATE STATUS
    // ALSO UPDATE LR
    if (word_count - last_word_count > 10000) {
      // UPDATE ACTUAL WORD COUNT BASED ON NEW INCREASE
      word_count_actual += word_count - last_word_count;
      // SET MOST RECENT SAVE
      last_word_count = word_count;

      // IF USER IS DEBUGGING REPORT STATUS
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }

      // UPDATE THE LR BASED ON WORD COUNT
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      // IF LR IS TOO SMALL THEN SET TO MIN THRESHOLD
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    // IF THERE IS NO SENTENCE LEN THEN ITERATE THRU 1K WORDS OR UNTIL WE HIT AN
    // EOF. AFTER OBTAINING SENTENCE, RESET SENTENCE POSITION TO 0
    if (sentence_length == 0) {

      // ITERATE THRU 1K WORDS OR UNTIL EOF. ADD THEM TO SENTENCE
      while (1) {
        // GET IDX OF WORD IN VOCAB
        word = ReadWordIndex(fi);
        // IF FILE HAS BEEN COMPLETED STOP LOOP
        if (feof(fi)) break;
        // IF LINE WASN'T WORD SKIP ITER
        if (word == -1) continue;
        // INCREASE WORD COUNT AS A WORD HAS BEEN FOUND
        word_count++;
        // IF WORD IS EOF WE CAN STOP
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        // USE SAMPLE PARAM, IF WORD OCCURS HIGHER THAN THE SAMPLE WE MAY RANDOMLY SKIP THIS WORD
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }

        // ADD CURRENT WORD IDX TO SENTENCE AND INCREASE SENTENCE LEN
        sen[sentence_length] = word;
        sentence_length++;

        // IF SENTENCE IS SIZE OF MAX WE CAN STOP
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }

      // RESET SENTENCE POSITION
      sentence_position = 0;
    }

    // IF EOF OR IF CURR WORD COUNT IS HIGHER THAN TRAINING WORDS PER THREAD
    // THEN TRAINING ITER DONE, IF ALL ITER COMPLETE END LOOPING
    // ELSE RESET WORD COUNTS AND SENTENCES AND JUMP BACK TO THREAD LOC IN FILE
    if (feof(fi) || (word_count > train_words / num_threads)) {
      // UPDATE THE ACTUAL WORD COUNT 
      word_count_actual += word_count - last_word_count;
      // WE HAVE COMPLETED A TRAINING ITER, DECREASE
      local_iter--;
      // IF LOCAL ITER IS 0 WE CAN STOP TRAINING
      if (local_iter == 0) break;
      // ELSE WE CAN RESET THE ITER WORD COUNT AND SENTENCE INFO
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      // SEEK THE LOCATION FOR THE THREAD IN THE FILE
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      // JUMP BACK TO START OF LOOP
      continue;
    }

    // SET WORD EQUAL TO CURR IDX DEFINED BY POS IN SENTENCE
    word = sen[sentence_position];
    // IF NOT AN IDX THEN JUMP TO START OF LOOP
    if (word == -1) continue;
    // FOR BOTH NEU1 AND NEU1E LAYERS RESET ALL VALS TO 0
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    // CALC RANDOM VAL
    next_random = next_random * (unsigned long long)25214903917 + 11;
    // TAKE THE NEXT RANDOM VAL AND PUT IT W/IN RANGE OF THE MAX WORD SKIP ALLOWED
    b = next_random % window;

    // IF MODEL IS USING CONTINUOUS BAG OF WORDS THEN CHECK THE WORDS
    // WITHIN THE GIVEN WINDOW AND PERFORM WEIGHT UPDATES. THIS CAN
    // BE DONE USING HIERARCHICAL SOFTMAX OR NEG SAMPLING
    // ELSE MODEL IS USING SKIP-GRAM
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      // SET CONTINUOUS WORD COUNT
      cw = 0;
      // FOR A MAX OF 2x THE WINDOW TAKE THE SURROUNDING WORDS AND
      // COPY THEIR HIDDEN LAYER INFO
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        // CALCULATE CURRENT POSITION IN SENTENCE
        c = sentence_position - window + a;

        // IF IDX ISN'T REAL WE CAN SKIP
        if (c < 0) continue;
        if (c >= sentence_length) continue;

        // SET THE LAST WORD TO THE CURRENT IDX
        last_word = sen[c];
        // IF WORD IDX ISN'T REAL SKIP
        if (last_word == -1) continue;

        // COPY LAYER INFO FOR THE GIVEN WORDS
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }

      // IF WE HAVE CONTINUOUS WORDS
      if (cw) {
        // NORMALIZE VALS BASED ON WORD COUNT
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;

        // IF HIERARCHICAL SOFTMAX CALC SIGMOID VAL FOR WORD
        // UPDATE LAYER WEIGHTS BASED ON RES
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          // SET FREQ TO 0
          f = 0;

          // PERFORM FREQ CALC TO GET SIGMOID VAL
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }

        // IF NEGATIVE SAMPLING SWEEP THRU NEG SAMPLES 
        // CALC SIGMOID FOR WORD AND UPDATE LAYER WEIGHTS
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          // IF FIRST WORD IN NEG SAMPLING WE KNOW ITS A MATCH
          // ELSE FIND A RAND WORD THAT ISN'T A MATCH
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            // CALC NEXT RANDOM VAL AND PICK THIS RANDOM WORD
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            // IF WORD ISN'T VALID OR IS CURR WORD, SKIP
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }

          // PERF FREQ CALC TO GET SIGMOID VAL
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          // PROPAGATE ERRS 
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          // LEARN WEIGHTS FOR NEG LAYER
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        
        // hidden -> in
        // FOR A MAX OF 2x THE WINDOW TAKE THE SURROUNDING WORDS AND
        // COPY THEIR HIDDEN LAYER INFO
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          // CALCULATE CURRENT POSITION IN SENTENCE
          c = sentence_position - window + a;

          // IF IDX ISN'T REAL WE CAN SKIP
          if (c < 0) continue;
          if (c >= sentence_length) continue;

          // SET THE LAST WORD TO THE CURRENT IDX
          last_word = sen[c];
          if (last_word == -1) continue;

          // COPY LAYER INFO FOR THE GIVEN WORDS
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  //train skip-gram
      // FOR A MAX OF 2x THE WINDOW TAKE THE SURROUNDING WORDS AND
      // COPY THEIR HIDDEN LAYER INFO
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        // CALCULATE CURRENT POSITION IN SENTENCE
        c = sentence_position - window + a;
        
        // IF IDX ISN'T REAL WE CAN SKIP
        if (c < 0) continue;
        if (c >= sentence_length) continue;

        // SET THE LAST WORD TO THE CURRENT IDX
        last_word = sen[c];
        // IF WORD IDX ISN'T REAL SKIP
        if (last_word == -1) continue;

        // SET LAYER ERR INFO TO NONE
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        
        // IF HIERARCHICAL SOFTMAX CALC SIGMOID VAL FOR WORD
        // UPDATE LAYER WEIGHTS BASED ON RES
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {

          // SET FREQ TO 0
          f = 0;

          // PERFORM FREQ CALC TO GET SIGMOID VAL
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }

        // IF NEGATIVE SAMPLING SWEEP THRU NEG SAMPLES 
        // CALC SIGMOID FOR WORD AND UPDATE LAYER WEIGHTS
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          // IF FIRST WORD IN NEG SAMPLING WE KNOW ITS A MATCH
          // ELSE FIND A RAND WORD THAT ISN'T A MATCH
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            // CALC NEXT RANDOM VAL AND PICK THIS RANDOM WORD
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            // IF WORD ISN'T VALID OR IS CURR WORD, SKIP
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          
          // PERF FREQ CALC TO GET SIGMOID VAL
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          // PROPAGATE ERRS 
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          // LEARN WEIGHTS FOR NEG LAYER
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }

    // MOVE TO NEXT WORD IN SENTENCE
    sentence_position++;
    // IF SENTENCE COMPLETED WE CAN JUMP BACK TO START OF LOOP
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

// TRAINING FXN FOR WORD2VEC
void TrainModel() {
  // a: USED TO ITER THRU THREADS
  // b: USED TO ITER THRU WEIGHTS
  // c: USED TO ITER THRU WEIGHTS PER CLASS
  // d: USED TO ITER THRU WEIGHTS PER CLASS
  long a, b, c, d;
  // DECLARE OUTPUT FILE
  FILE *fo;
  // SETUP PTHREADS TO THE # OF REQUESTED THREADS
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  // SET INITIAL LR
  starting_alpha = alpha;
  // IF GIVEN A PRE-LEARNED VOCAB, ADD IT TO VOCAB, VOCAB HASH, SET TRAINING WORD
  // COUNT REMOVE LOW COUNT WORDS
  // ELSE CREATE VOCAB FROM TRAIN SAMPLE
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  // IF USER DESIRES TO SAVE VOCAB STORE VOCAB TO DESIRED FILE
  if (save_vocab_file[0] != 0) SaveVocab();

  // IF USER DIDN'T TELL US WHERE TO SAVE WORD VECTS RETURN
  if (output_file[0] == 0) return;

  // ASSIGN HUFFMAN CODES TO VOCAB, INIT NETWORK LAYERS
  InitNet();

  // IF USING NEG SAMPLING CREATE UNIGRAM TABLE FOR QUICK SAMPLING TO FIND NEGATIVES
  if (negative > 0) InitUnigramTable();
  
  // START TIMER FOR TRAINING
  start = clock();

  // FOR ALL AVAIL TREADS BEGIN MODEL TRAINING
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  // WAIT FOR ALL THREADS TO COMPLETE
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  // OPEN OUTPUT FILE TO SAVE WORD VECTS
  fo = fopen(output_file, "wb");
  // IF SAVING INFO AS VECTOR FORMAT PROCESS IN BINARY OR TEXT
  // ELSE SAVING IN CLASSES FORMAT
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    // FOR EACH WORD PRINT THE WORD AND PRINT THE WEIGHTS ASSOC W/ WORD
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

// SEARCH THRU ARG SPACE FOR VAR PASSED IN
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

// MAIN FXN FOR WORD 2 VEC
// SETS MODEL TO SKIPGRAM OR CBOW, ACCEPTS ARGS FOR MODEL TRAINING
// INIT VOCAB STORAGE, INIT VOCAB HASH SETUP SIGMOID TABLE
int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  // SIZE OF WORD VECTS
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  // FILE TO USE FOR TRAINING
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  // FILE LOCATION TO SAVE VOCAB
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  // READ VOCAB FROM FILE, DON'T TRAIN 
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  // TURN ON DEBUG TO GET INFO DURING TRAINING  
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  // TOGGLE FILE SAVING FORMAT 
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  // TOGGLE BETWEEN CBOW & SKIP-GRAM 
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  // DEFAULT LR FOR CBOW  
  if (cbow) alpha = 0.05;
  // SET LR  
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  // WHERE TO SAVE WORD VECTS  
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  // MAXIMUM SKIP LEN BETWEEN WORDS 
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  // SET THRESHOLD FOR WORD APPEARANCE
  // IF OCCURS MORE THEN MIGHT BE RANDOMLY DOWN SAMPLED 
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  // USE HIERARCHICAL SOFTMAX  
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  // NUMBER OF NEG SAMPLES TO USE  
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  // THREADS AVAIL TO PROG  
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  // TRAINING ITER COUNT  
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  // DISCARD WORDS THAT OCCUR LESS THAN  
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  // USE WORD CLASSES INSTEAD OF VECTS  
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  // ALLOC MEM TO STORE VOCAB WORD STRUCT BASED ON CURR MAX SIZE OF ARR
  // THE MAX SIZE DEFAULT IS 1000 ENTRIES
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  // SETUP HASH TAB TO STORE VOCAB, MAX 21M
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  // TABLE USED FOR SIMULATING SIGMOID FXN, SPEEDS UP CALCS
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  // FILL SIGMOID TABLE
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  // BEGIN TRAINING
  TrainModel();
  return 0;
}

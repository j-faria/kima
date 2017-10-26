// are there observations after the HARPS fiber change?
#define obs_after_fibers true

// use hyper parameters muP, wP, and muK 
#define hyperpriors false


#define ananas true
#define maracuja false
#define limao false

#if ananas
    #define DOCEL false
    #define GP false
#elif maracuja
    #define DOCEL false
    #define GP true
#elif limao
    #define DOCEL true
    #define GP true
#endif

#define trend false

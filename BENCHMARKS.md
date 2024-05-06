vLLM-inference-AQLM
===
Model Inferencing Speed Benchmarks

## Hardware
3090TI 24GB VRAM (had to exit xwindows to get enough VRAM as it uses *everything*)

#### Llama-3-8B
```python
llm = LLM(
      model="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
      enforce_eager=True,
      gpu_memory_utilization=0.99,
      max_model_len=8192, # max size for this particular model
)
```

```bash
¡Con gusto! Here's a poem about the sun in Spanish:

Sol de oro, de luz me envuelve,
En cada momento, me ilumina con calidez,
Con tus rayos, me acaliento, me cálculo,
Y me guía en el camino, en cada momento.

Tu faz es el sol, que me hace sentir,
Que todo es posible, todo es real,
Que no hay nada que no pueda hacer,
Y que mi vida es una eternidad, en tu luz.

Sol de mi vida, sin ti no soy,
Un ser sin luz, sin ti no soy,
Y sin ti, el mundo es una oscuridad,
Y sin ti, no hay vida, no hay vida.

¡Sol de mi vida, te amo tanto!
¡Sol de mi vida, que no te deje ir!
¡Sol de mi vida, que siempre estés conmigo,
Y que ilumines mi camino, y me guíes!

Translation:

Golden sun, you envelop me with light,
In every moment, you illuminate me with warmth,
With your rays, you calm me, you measure,
And guide me on the path, in every moment.

Your face is the sun, that makes me feel,
That everything is possible, everything is real,
That there's nothing that I cannot do,
And that my life is an eternity, in your light.

Sun of my life, without you, I am not,
A being without light, without you, I am not,
And without you, the world is darkness,
And without you, there is no life, no life.

Oh, sun of my life, I love you so much!
Oh, sun of my life, don't leave me!
Oh, sun of my life, always be with me,
And guide me on my path, and lead me!

Note: This is a personal poem, so it's not a traditional or classical poem, but it's a creative expression of the poet's feelings and thoughts about the sun.

===
Generated 419 tokens in 6.77 seconds = 61.92 tok/sec
```

#### Llama-3-70B
```pyton
llm = LLM(
      model="ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16",
      enforce_eager=True,
      gpu_memory_utilization=0.99,
      max_model_len=5120, # this is the context size
      kv_cache_dtype="fp8", # enabling this allows ~5k context, without it max is ~3k
)
```

```bash
Here is a poem about the sun in Spanish:

El sol, rey del cielo azul,
Despierta la vida con su calor,
Ilumina la tierra con su luz,
Y nos da energía con su fulgor.

Con sus rayos dorados y brillantes,
Tibera la niebla y el frío,
Y nos muestra el camino recto,
Para encontrar la felicidad.

En su camino diario, nos habla,
De la esperanza y del amor,
De la vida y de la muerte,
Y de la eternidad que nos rodea.

El sol, fuente de vida y de luz,
Nos enseña a vivir con alegría,
A apreciar cada momento,
Y a disfrutar de la vida con intensidad.

Translation:

The sun, king of the blue sky,
Awakens life with its warmth,
Illuminates the earth with its light,
And gives us energy with its glow.

With its golden and brilliant rays,
It dispels the fog and the cold,
And shows us the straight path,
To find happiness.

In its daily journey, it speaks to us,
Of hope and love, of life and death,
And of the eternity that surrounds us.

The sun, source of life and light,
Teaches us to live with joy,
To appreciate every moment,
And to enjoy life with intensity.

===
Generated 289 tokens in 36.04 seconds = 8.02 tok/sec
```
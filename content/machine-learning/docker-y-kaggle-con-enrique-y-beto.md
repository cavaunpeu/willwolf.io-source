Title: Docker y Kaggle con Enrique y Beto
Date: 2017-03-22 19:56
Author: Will Wolf
Lang: es
Url: 2017/03/22/docker-y-kaggle-con-enrique-y-beto/
Save_as: 2017/03/22/docker-y-kaggle-con-enrique-y-beto/index.html
Slug: docker-y-kaggle-con-enrique-y-beto
Status: published
Summary: Este post tiene como objetivo familiarizarlos con lo que es Docker, por qué y cómo usarlo para Kaggle.
Image: ../images/ernie_and_bert.png

Este post tiene como objetivo familiarizarlos con lo que es Docker, por qué y cómo usarlo para Kaggle. Para hacer las cosas más simples, hablaremos principalmente de Plaza Sésamo y pasteles en lugar de computadoras y datos.

Una mañana de lunes, Enrique sale de debajo de su cobija rayada, pone los dos pies en el piso y abre la ventana de su cuarto. Echa un vistazo hacia la metrópoli llena de galletas y muñecos de peluche, endereza el cuello de su suéter color banana, suelta un fuerte bostezo matutino y exclama: "Hoy, voy a preparar pasteles para mi querido compañero Beto."

![ernie and bert]({static}/images/ernie_and_bert.png)

Por mala suerte, Enrique nunca ha hecho pasteles antes. ¡Pero no importa! Se precipita con prisa hacia la cocina, toma un libro de cocina, organiza los ingredientes y prende su hornito Easy-Bake. "Lo pongo a prueba igual. Prepararé el mejor pastel jamás hecho por todos los muñecos de peluche. Y una vez que el resultado me complazca, haré 50 más," grita.

Horas después, su trabajo termina: su pastel - tres pisos de "mini-pasteles" sabor arándano, fresa y tocino - es simplemente la mejor cosa que haya probado alguna vez en su vida. "¡Mucho mejor que cualquier cosa que ese fraude Cookie Monster haya probado jamás!" dice. Emocionado, Enrique toma una pausa en su cocina ya demasiada sucia para admirar el resultado. Piensa en Beto y se pregunta qué tan rápido se puede entregar el regalo. "Ya que he horneado el pastel perfecto, solo me toca hacer 50 más. ¿Debe ser fácil, verdad?"

Enrique se da vuelta para mirar su Easy-Bake. "Pues, esa vaina solo cocina un pastel a la vez. ¡Tardaría días en hacer 50 a ese ritmo!" Aún de buen humor, corre a la panadería del pueblo para pedir prestado su horno - ese más grande que el suyo. Se lo dan en seguida y Enrique se pone a cocinar.

Desafortunadamente, mientras está mezclando los ingredientes empieza a tener problemas con las herramientas de la panadería. El mezclador eléctrico se rompe. El cuchillo no corta las fresas de la manera correcta. Las tazas de medir tienen tamaños sutilmente distintos. Enrique se estresa. Pensaba que estaba al punto de terminar, pero se da cuenta de repente de que en realidad acaba de empezar. Aunque vino con la receta exacta en mano, se nota que ahora está usando herramientas diferentes en una cocina extraña, en un ambiente nuevo. "¿No puede un muñeco de peluche cocinar un solo pastel en su horno pequeño, luego traer la receta y los ingredientes a un horno más grande y ahí fabricarlos rápidamente a escala masiva? ¿Por qué tiene que ser tan complicado esto?"

## Presentando Docker

Con tristeza y desesperación, Enrique camina al puerto para aclarar su mente. Allí, se encuentra con cientos de contenedores azules y blancos del tamaño de camionetas y se le ocurre una idea divertida: "¿Qué tal si cocino allí? Moveré todas mis herramientas dentro del contenedor - la tabla de cortar, el cuchillo, el mezclador, los utensilios - y escribiré la receta en el muro interior. La única cosa que faltará sería el horno, pero eso se obtiene en todas partes. Así, usando el horno en mi casa, puedo continuar horneando un pastel a la vez como siempre; por el contrario, usando el horno de la panadería puedo cocinar de capacidad aumentada. Listo. Enrique agarra el primer contenedor que ve y corre a casa para llenarlo de pastel."

Después de escribir la receta en el muro interior del contenedor, Enrique se da cuenta de que lo que quiere traer a la panadería tiene que ser ligero. Si no, ¡no lo podrá llevar! Por lo tanto, en lugar de físicamente llevar sus herramientas - el cuchillo, el mezclador, etc. - simplemente escribe los nombres y números de estos productos además de instrucciones para adquirirlos por fuera. De la misma manera, en lugar de encerrar los ingredientes mismos, espera que estén disponibles en la panadería una vez que llegue. Así, cuando la receta diga "echa 3 cucharadas de azúcar del gabinete," el azúcar ya estará puesto en el gabinete mismo.

## Presentando Kaggle

Hornear pasteles en Plaza Sésamo es una metáfora de construir modelos para Kaggle. Típicamente, construimos prototipos sencillos en nuestra entorno local y luego alquilamos una máquina más poderosa ubicada en alguna granja en Virginia para hacer el trabajo pesado en el sentido computacional. En las competencias de Kaggle, el problema inicial de Enrique es demasiado común: aún después de lograr conseguir un mezclador eléctrico, tazas de medir, etc. que se parecen a los suyos - esto es, aún después de instalar todas aquellas librerías en el servidor remoto que teníamos en el local - los entornos aún no eran idénticos y algunos problemas surgen en seguida. Los contenedores de Docker resuelven este problema: si podemos lograr hornear pasteles una sola vez en nuestra cocina, podemos así rehacerlos de manera determinista *n* veces en cualquier cocina de fuera - y preferiblemente en una con un horno mucho más poderoso que el nuestro.

## Con ustedes, los servidores remotos

Un servidor remoto es la panadería: es una computadora como la nuestra, pero que puede procesar datos más rápido y en cantidades más grandes. En otras palabras, es una cocina con un horno más grande.

## Utensilios de cocina y los ingredientes

En lugar de incluir utensilios de cocina en nuestro contenedor simplemente pormenorizamos cuáles necesitamos y cómo adquirirlos. Para una competencia de Kaggle, esto es igual a instalar las librerías - pandas, scikit-learn, etc. - necesarias para la tarea a mano. Una vez más, no tenemos que incluir estas librerías en nuestro contenedor, sino disponer de instrucciones para dónde y cómo instalarlas. En la práctica, esto se manifiesta como un `pip install -r requirements.txt` en nuestro [`Dockerfile`](https://docs.docker.com/engine/reference/builder/).

En lugar de incluir los ingredientes en nuestro contenedor asumimos que estarán disponibles en la panadería anfitriona. Esto es un poco más complicado de lo que suena por las siguientes razones:

1. Nuestra panadería anfitriona está a un par de cuadras de nuestra casa. Si queremos que estén disponibles los ingredientes en esa panadería, tenemos que traerlos ahí físicamente de alguna manera.
2. Aún después de traerlos físicamente a la panadería, el cocinar que resulte dentro del contenedor estará aislado del resto de la panadería misma: la única cosa con la que se conecta exteriormente es su horno.

Para una competencia de Kaggle, ¿cómo hacemos que los datos locales sean utilizables *dentro del contenedor, hospedado en un servidor remoto*?

### Los "Docker Volumes"

Los "Docker Volumes" permiten que datos sean compartidos entre una carpeta dentro de un contenedor y una carpeta en el sistema de archivo local del servidor hospedando ese contenedor. Esto es igual a lo que ocurre cuando:

1. Enrique trae sus ingredientes a la panadería, junto con (pero no dentro de) su contenedor.
2. Al llegar, pone un tarro de azúcar en un cubo azul en la esquina de la sala.
3. Se estipula que, al comenzar a hornear dentro del contenedor de la la panadería, los ingredientes se deberían compartir entre el cubo azul en la esquina de la sala y el gabinete. Así, cuando la receta diga "agarra un tarro de azúcar del gabinete," Enrique puede extender la mano hacia el gabinete dentro del contenedor y recuperar el tarro de azúcar del cubo azul en la esquina de la panadería. Recuerden: el contenedor no vino con ningún ingrediente empacado por dentro; el gabinete hubiera estado vacío por la misma razón.

Trayendo el contenedor a la panadería es igual a un sencillo `docker run` con el servidor remoto como el `docker-machine`. Trayendo los ingredientes a la panadería, esto es colocando datos en el sistema de archivo local del servidor remoto, es mucho menos "sexy." En el sentido más simple, es igual a usar `scp` or `rsync` para transferir un archivo del entorno local al servidor remoto hasta usar `curl` para descargar un archivo directamente en el servidor remoto mismo.

En la práctica, esto se ve generalmente así:

```
docker
    --tlsverify
    --tlscacert="$HOME/.docker/machine/certs/ca.pem"
    --tlscert="$HOME/.docker/machine/certs/cert.pem"
    --tlskey="$HOME/.docker/machine/certs/key.pem" -H=tcp://12.34.56:78
run
    --rm
    -i
    -v
    /data:/data kaggle-contest build_model.sh
```

## Utensilios de cocina que no se compran en la tienda

Para hacer su pastel, Enrique usó una tabla de cortar única en el mundo que Beto hizo a mano para él. ¿Cómo puede usarla en la panadería? En términos de Kaggle: ¿cómo puedo usar una librería en mi proyecto que no está disponible en un repositorio de paquetes público (una que construí yo mismo)?

Para esto, no existe una "fórmula secreta." Con la tabla de cortar/librería, podemos:

1. Incluirla en el contenedor y aguantar el peso extra.
2. Tratarla como ingrediente, traerla a la panadería y accederla vía un "Docker Volume."

## Feliz cocinada

Trasladar tu entorno local al interior de un contenedor de Docker, y/o "Dockerizar" este entorno una vez que estés listo para usar un servidor remoto para hacer el trabajo pesado, asegurará que solo tendrás que averiguar cómo hacer el pastel una sola vez. Haz tus prototipos localmente, luego enviarlos sin estrés a la panadería para la producción de escala masiva.

¡Muy buen provecho!

---
Recursos adicionales:

Aquí dejamos dos recursos adicionales que creo útiles para aprender
sobre Docker para Kaggle:

1.  [Workflow, Serialization & Docker for Kaggle](https://speakerdeck.com/smly/workflow-serialization-and-docker-for-kaggle)
2.  [How to get started with data science in containers](http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/)

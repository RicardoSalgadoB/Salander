# Salander
This is the official repository for the participation of our team in the NASA Space Apps Global Hackathon. Wish us luck.

Documentation (with images):  
[[https://drive.google.com/file/d/10as3Vw_5V0Yt2NgQPKuI-C1uO0wsSY2F/view?usp=sharing]](https://drive.google.com/file/d/10as3Vw_5V0Yt2NgQPKuI-C1uO0wsSY2F/view?usp=sharing)

Demo: 
[[https://drive.google.com/drive/folders/1TzMc5W7SJszV9PoLHWGMmOJ0rg9KGKJj]](https://drive.google.com/drive/folders/1TzMc5W7SJszV9PoLHWGMmOJ0rg9KGKJj)
[[https://salander-test.web.app]](https://salander-test.web.app)

### High-level Summary

The core of our project is a multi-algorithm platform that can adapt to the restraints of the
infrastructure and the needs of the researcher. Through this platform, the researchers can ask
the automated seismic station to perform a specific algorithm, based on the computational
power available and the desired accuracy in the data. Each algorithm is capable of analyzing the
data on its own, but each has its tradeoffs. For example, the statistical-only algorithm we’ve
crafted requires littles computational power, but lacks adaptability. The STA-LTA (Short-term
average, long-term average) local maxima algorithm offers a compromise between both. While
the CNN (Convolutional Neural Network) model is certainly on the higher end of computational
power. Though this can certainly be confusing and overwhelming for a potential user, we have
implemented an LLM (Large Language Model) through the Gemini API to guide the user through
the implications of choosing x or y algorithm. Then, the capabilities of the platform could
gradually turn into a hub for learning and for learning to analyze this sort of data. Even on the
higher end of Academia, our platform could help to streamline research and lead into
uncountable scientific discoveries.

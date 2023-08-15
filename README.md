# Shot Type Classification for Ads


<ins>**Context:**

Creating a deep learning model to analyze brand assets on images and video to predict what the audience will remember and pay attention to, providing brands an understanding of which ads will have higher memorability and increase brand recall and awareness among consumers.   
This same technology is widely used in the advertisement industry (e.g YouTube) to identify opportune places to insert ads in the user-created content and correlate the effectiveness of the ads.

---
<ins>**Scope of this project:**

Many attributes can be analyzed in an image and video ad such as saliency, position, colors and shapes, objects, scenes, the relation among them, etc.
In this case, we will be training our model to classify 2 Shot Types:

1. Shot Scale (long shot, full shot, medium shot, close-up shot, extreme close-up shot)
2. Shot movement (static shot, motion shot, zooms in for push shot, and zooms out for pull shot)

---
<ins>**Proposed Architecture:**

* Input: video --> will be dissected into frames  
* Feature extraction: with InceptionV3 model  
* Output:  
    * Shot Scale Class
    * Shot Movement Class

---

---
<ins>**Worth mentioning:**

* Deeper dive below the hood of Keras
    * More flexibility and capability
---




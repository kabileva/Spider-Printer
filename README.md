## Demo:

**Demo video: https://youtu.be/3viI6UhH82M**

# Spider Printer
Implementation of "A Computational Approach for Spider Web Inspired Fabrication of String Art"(Seungwoo Je, Yekaterina Abileva, Andrea Bianchi, Jean-Charles Bazin. 2019). In Computer Animation and Virtual Worlds. https://onlinelibrary.wiley.com/doi/full/10.1002/cav.1904

Creating objects with threads is commonly referred as string art. It is typically a manual, tedious work reserved for skilled artists. In this paper, we investigate how to automatically fabricate string art pieces from one single continuous thread in such a way that it looks like an input image. The proposed system consists of a thread connection optimization algorithm and a custom-made fabrication machine. It allows casual users to create their own personalized string art pieces in a fully automatic manner. Quantitative and qualitative evaluations demonstrated our system can create visually appealing results.

## Scripts:
 - Local cost algorithm - LocalCostAlgorithm.py
 - Global cost algorithm - GlobalCostAlgorithm.py

**For 3D:**
 - run 3dpins.py
 - Change Dimension to '3D' in GlobalCostAlgorithm.py

**Hardware:**
 - Laser cut layots can be found in hardware folder
 - Processing code for controlling hardware - in hardware/MotorControl
 
**Simulated Annealing implementation:**
 - simulatedAnnealing.py </br>
 *ref.: Wai-Man Pang, Yingge Qu, Tien-Tsin Wong, Daniel Cohen-Or, and Pheng-Ann Heng. 2008.
Structure-aware halftoning. ACM Transactions on Graphics (SIGGRAPH) 27, 3 (2008), 89:1-89:8.*
 


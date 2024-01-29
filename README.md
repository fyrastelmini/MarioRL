# MarioRL (still under construction)
Extracts the player inputs from VOD of speedruns, requires a VOD with inputs shown like this:
![alt text](https://github.com/fyrastelmini/MarioRL/blob/main/Controller.png?raw=true) 

Currently works only for NES

Make dataset:
```python
python frames_extractor.py --youtube_link https://www.youtube.com/watch?v=TJnk7a-Lefo --start 12 --end 1160

# Rectangle recognition in printscreen from software with opencv2 Python library

## Purpose
It is proof of concept, that elements of a program can be recognized and defined as a list of coordinates for UI Path robot, to click in defined ared, and put data alsewhere.
Script has to recognize fields in windows (sashes) - rectangles - and return list of coordinates.

## Background
There was an idea to automatize one of the procesess with data input. Software where data has to be inserted is quite old, not responsive, doesn't allow to use batch data and was bottleneck of process. "If could know where the sash is, then we can click it, and then insert data elsewhere". Can we? That was a challenge, something fresh and juicy. I've heard about shape recognition in python, but never tried it, sounds like a great opportunity to learn something new. 
I've got only single screenshot and rough knowledge what should be recognized, what should be output. As this software was "from my yard" I knew what I was looking for.<br>
![Original image](/original_screenshot.jpg)
## How should it work?
Algorithm of whole process looks like this:
{...}-> robot save screenshot->script recognize sashes-> returns coordinates-> robot click at coordinates and put data{...}

## Sources

### First look
This was completly new subject for me, so first of all I've tried to watch few youtube videos to check what exactly I'm looking for.
Phrase "shape recognition python" should take you to most updated list. I've watched few of them and knew that I should focus on detecting rectangles.

### First script
In first approach I've used: [Detect rectangles in OpenCV 4.2.0 using Python 3.7](https://stackoverflow.com/questions/61166180/detect-rectangles-in-openc) as a starting point. 
But the results was not good enough for me, it was not easy to find all rectangles I need, or only those which I need. So I've tried another approach. 

### Second approach
[Find and Draw Contours with OpenCV in Python](https://thinkinfi.com/find-contours-with-opencv-in-python/)<br>
This is second tutorial which I've found very helpful in my case, it goes step by step using simple image to find all the shapes:<br>
![Original input image](https://um0ec8.p3cdn1.secureserver.net/wp-content/uploads/2021/11/Read-input-image-for-contour-detection-opencv-python-768x598.png)<br>
after all steps all shapes has been recognized and marked:<br>
![Output image](https://um0ec8.p3cdn1.secureserver.net/wp-content/uploads/2021/11/opencv-draw-contours-bounding-box-for-all-contours-python-768x613.png)<br>
tutorial has full code listed, what I've done I've adjusted roi points, focused only on rectangles, and add some code to return coordinates. 
After that script is listing coordinates:<br>
`shape no:  1`<br>
`x start:  8`<br>
`y start:  88`<br>
`width : 452`<br>
`height : 202`<br>
as well as (for visual check only) mark recognized rectangles (bright green):<br>
![Screenshot with detected contour](/detected_contour.png)

There are some hardcoding made in script to find optimal way of finding sashes and only sashes. 
To remove hardcoding, more screenshots is needed. 

### To do
[ ] remove hardcoding (as much as possible)<br>
[ ] add auto threshold recognition<br>
[ ] characters recognition (if sequence of sashes is not very obvius)<br>

## Summary
As proof of concept script is succesful. It recognizes sashes and only sashes. Without more screenshots it is difficult to find which part of script need to be adjusted.
At this point automation team has decided that we will not follow this path (of recognizing sashes from printscreen) due to high complication and possible errors. As a suggestion we left project to change original software to allow scripting or batch change of data.
For me it was fun and challenging project to learn something new and try different approaches to problems.
Hope you will find it usefull for your own projects.

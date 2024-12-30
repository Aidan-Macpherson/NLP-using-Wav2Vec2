# NLP-using-Wav2Vec2
This repository codes a natural language processor using Meta's Wav2Vec2 self-supervised model for automatic speech recognition.

The python script is labeled, "NLP Python Script." It takes in an audio wav file and utilizes the Wav2Vec2 pre-trained tokenizer and model to form a speech-to-text prediction, whose output is printed.

The wav file provided, labeled "OSR_us_000_0010_8k," is taken from the Open Speech Repository. It is loaded in using librosa.load, but make sure to change the path to its proper location in your directory. The actual text of the audio file is as follows: 

"The birch canoe slid on the smooth planks.
Glue the sheet to the dark blue background.
It's easy to tell the depth of a well.
These days a chicken leg is a rare dish.
Rice is often served in round bowls.
The juice of lemons makes fine punch.
The box was thrown beside the parked truck.
The hogs were fed chopped corn and garbage.
Four hours of steady work faced us.
A large size in stockings is hard to sell."

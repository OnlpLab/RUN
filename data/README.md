# OpenStreetSet (OSS) Data

The OSS corpus contains 4 files each appearing in the format .pickle (using Python 3.6.1):

  - grids (tiles)
  - maps
  - dataset
  - valselct

### grids (tiles)

  - maps coordinates boarders in the map to the grid-id and x and y graph.
 
### maps 
  - contains 3 maps from OpenStreetMap. Each map contains the following attributes: 
  
    | attriute | description |
    | ------ | ------ |
    | grids | a list of all grids |
    | grids_to_streets | a list of all the grids and for each grid there is a list of streets that intersect with this grid. |
    | streets | a list of all streets in the map. Each street consists of the following attributes: (1) a list of ordered grids; (2) street-id; (3) street-name| 

### dataset
  - the data collected using Amazon Mechanical Turk (MTurk). 
  - The data consists is divided into 3 maps.
  - Each map contains a list of all the examples collected in that map.
  - Each example holds the following data: 
      
    | attriute | description |
    | ------ | ------ |
    | id | id of the navigation story. |
    | line_number | order of the instruction in the story. |
    | instruction | the sentence as given by the instructor.| 
     | actions | list of actions that follow the route path.|
    | route | a list of points constituting the route path. It is produced by executing the actions one by one.|
    | list_of_words | a list of the word-token in the sentence. |
    | list_of_words_with_entity_extraction | the list of word-token after pre-process of entity extraction layer.|
    | map_variable_to_entity_streets | a map of the connections between the variable of type street and the original entity name.|
    | map_variable_to_entity_not_streets | a map of the connections between the variable of type 'other than street' and the original entity name.| 
    | actions_with_face_and_verifier* | list of actions that follow the route path. It uses an addition of two actions: 'FACE' and 'VERIFY'. (not used in the model)| 
    | route_with_face_and_verifier* | list of points constituting the route path. It is produced by executing the actions one by one with an addition two actions: 'FACE' and 'VERIFY'. (not used in the model)| 

			
*OSS has a variation that contains two more types of actions: 
  - 'FACE': is a change of direction to face a specific street and end of street. 
  - 'VERIFY': gives verification to the current direction when it is explicitly mentioned in the instructions. For  example, 'turn right on to 8th Avenue'.
  
 However, we have been unsuccessful in benefiting from 'FACE' and 'VERIFY'. We leave them for future research.

### valselct
  - indicates the samples used for validation only in each map



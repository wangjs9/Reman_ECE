* This folder contains the following files:
 - reman-version1.0.xml: the corpus itself
 - reman-schema.xsd: XSD schema
 - reman.dtd: DTD definition
 - guidelines.pdf: official annotation guidelines
 - license.txt: full text of license information


* The REMAN (Relational EMotion ANnotation) corpus provides the annotation of emotions with the roles of experiencer, cause, and target in 1720 sentence triples subsampled from the Project Gutenberg (www.gutenberg.org).

XML file structure:

1. "adjudicated" node:

Includes annotations accepted by the curator. The node includes two layers: "spans" and "relations". "spans" include information about the annotations of phrases. "relations" include information about the annotation of relations.

2) "other" node is structured identically to "adjudicated" node but includes annotations that were rejected by the curator.


* Please cite the corpus as:
@inproceedings{Kim2018,
  author = {Kim, Evgeny and Klinger, Roman},
  title = {Who Feels What and Why? Annotation of a Literature Corpus with Semantic Roles of Emotions},
  booktitle = {Proceedings of COLING 2018, the 27th International 
      Conference on Computational Linguistics},
  month = {August},
  year = {2018},
  address = {Santa Fe, USA},
}

* The corpus can be found at: http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/reman.html


* The REMAN corpus is licensed under Creative Commons 4.0 license (https://creativecommons.org/licenses/by/4.0/). See file license.txt for more information.


* NOTE: Our license does not include Project Gutenberg's license. 


* The current version of the corpus is 1.0


* Contact information: Evgeny Kim evgeny.kim@ims.uni-stuttgart.de
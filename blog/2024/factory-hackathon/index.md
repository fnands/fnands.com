---
layout: post
title: Building your own personal ghostwriter
subtitle: "\n"
description: A simple hackathon project to build a personal ghostwriter using Mistral's API. 
categories: [mojo]
date: "2024-10-03"
author: "Ferdinand Schenck"
draft: true
---

This past weekend I participated in a AI hackathon organized by Factory Network and {Tech: Berlin} with my friends Axel Nordfeldt and [Jonathan Nye](https://denyed.xyz/). The sponsors were [Mistral](https://mistral.ai/), [Weaviate](https://weaviate.io/) and [LumaAI](https://lumalabs.ai/), meaning we had a bunch of fun credits to play around with. 
We ended up making it to the final six teams with our project, which was a pretty cool experience.

We spitballed a few ideas the week before the event, but eventually settled on building a kind of ghostwriter during the event. The idea wasn't to get chatbot to write you a story based on a minimal prompt, but rather to have something that interviews you in depth, and uses the interview as the basis of a blog post, short story or article.   

The idea was inspired by an experience Axel had where he went on a trip and later recounted it to a friend who is a talented writer. The friend then wrote up a summary of the trip and gave it to Axel, who was impressed by how well the friend had captured the essence of the trip. 

Writing about my own travel experiences is something I also wish I could do better. Not to impress anyone else, but just for my own memories. I went to India in 2012 and I have a handful of photos of the trip, but I didn't write much down at the time. I'd love to have a more detailed account of the trip, but I can't remember all the details anymore. I recently found a [blog post](https://travelingwithbangs.wordpress.com/2012/11/05/dogs-in-darjeeling-im-jane-goodall-or-something/) by someone I had met on the trip that even includes a quote from me, which I hardly remember.   

I struggle with journaling or keeping diaries in general, so I would like to have a tool that can help me write about my experiences. I don't know if it is just because writing is hard (part of the reason I keep this blog is to practice writing), or if it is just because I feel a bit silly writing about myself. 

What isn't that hard however is telling someone about my experiences, especially someone who knows how to ask pertinent questions. So the idea was to build a tool that could ask you questions about a experience, and then write up a summary based on your answers. Effectively the way a ghostwriter would work, but instead of writing a book for you, it would write a blog post or short story.

## The build

For the hackathon we had access to the sponsors' APIs. We especially wanted to play around with Mistral's API, which had just recently added a new model, [Pixtral](https://mistral.ai/news/pixtral-12b/) that can understand images. 

So we basically built the app (called JournalAIst) as two parts: an interviewer and a writer. Basically we had a chat loop in which you could upload images (interpreted by Pixtral) and during which the interviewer model would ask you questions about your experience and the images that you added. After answering the interviewers questions for a while, you can choose to end the conversation, after which you are sent to the writer. 

![Interviewer](interviewer_interface.png)

The writer will then be given the transcript of your conversation, as well as Pixtral's descriptions of any images you might have uploaded. The model then uses this context to write a few paragraphs worth of text about your experience. You get to choose if you want a blog post, short story or article, each of which having a slightly different tone and point of view. 

![Writer](writer_interface.png)


We had a little bit of time left at the end so we ended up also adding a section where we fed a summary of your post to LumaAI's Dream Machine model, which generates a five second video based on this input. 
<video src="https://github.com/user-attachments/assets/42943e30-7969-449a-bfaf-b60d463b4e3e" controls="controls" width="100%"> </video>



##



[Kruger Adventure](kruger_adventure.md)


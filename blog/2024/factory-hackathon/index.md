---
layout: post
title: Building your own personal ghostwriter
subtitle: "\n"
description: A simple hackathon project to build a personal ghostwriter using Mistral's API. 
categories: [llms]
date: "2024-10-03"
author: "Ferdinand Schenck"
draft: false
---

This past weekend I participated in a AI hackathon organized by [Factory Network](https://factory.network/) and [{Tech: Berlin}](https://www.techberlin.io/) with my friends Axel Nordfeldt and [Jonathan Nye](https://denyed.xyz/). The sponsors were [Mistral](https://mistral.ai/), [Weaviate](https://weaviate.io/) and [LumaAI](https://lumalabs.ai/), meaning we had a bunch of fun credits to play around with. 

I'm an ML engineer, but I haven't had that much exposure to the world brave new world of AI engineering that consists of calling LLM API's and prompt engineering, so I wanted to use the hackathon as a way to get a better understanding of which tools are out there are how people are using them. 

We spitballed a few ideas the week before the event, and eventually settled on building a kind of ghostwriter during the event. The idea wasn't to get chatbot to write you a story based on a minimal prompt, but rather to have something that interviews you in depth, and uses the interview as the basis of a blog post, short story or article.   

The idea was inspired by an experience Axel had where he went on a trip and later recounted it to a friend who is a talented writer. The friend then wrote up a summary of the trip and gave it to Axel, who was impressed by how well the friend had captured the essence of the trip. 

Writing about my own travel experiences is something I also wish I could do better. Not to for the sake of anyone else, but just for my own memories. I went to India in 2012 and I have a handful of photos of the trip, but I didn't write much down at the time. I'd love to have a more detailed account of the trip, but I can't remember all the details anymore. I recently found a [blog post](https://travelingwithbangs.wordpress.com/2012/11/05/dogs-in-darjeeling-im-jane-goodall-or-something/) by someone I had met on the trip which even includes a quote from me, which I hardly remember.   

I struggle with journaling or keeping diaries in general, so I would like to have a tool that can help me write about my experiences. I don't know if it is just because writing is hard (part of the reason I keep this blog is to practice writing), or if it is just because I feel awkward writing about myself. 

What isn't that hard however is telling someone about my experiences, especially someone who knows how to ask pertinent questions. So the idea was to build a tool that could ask you questions about an experience, and then write up a summary based on your answers. Effectively the way a ghostwriter would work, but instead of writing a book for you, it would write a blog post or short story.

## The build

For the two-day hackathon we had access to the sponsors' APIs. We especially wanted to play around with Mistral's API, which had just recently added a new multi-modal model, [Pixtral](https://mistral.ai/news/pixtral-12b/) that can understand images as well as text. 

So we basically built the app using Streamlit (called JournalAIst) as two parts: an interviewer and a writer. Basically we had a chat loop in which you could upload images (interpreted by Pixtral) and during which the interviewer model (`mistral-large-2407`) would ask you questions about your experience and the images that you added. After answering the interviewers questions for a while, you can choose to end the conversation, after which you are sent to the writer. 

![Interviewer](interviewer_interface.png)

The writer will then be given the transcript of your conversation, as well as Pixtral's descriptions of any images you might have uploaded. The model then uses this context to write a few paragraphs worth of text about your experience. You get to choose if you want a blog post, short story or article, each of which having a slightly different tone and viewpoint. 

![Writer](writer_interface.png)


We had a little bit of time left at the end so we ended up also adding a section where we fed a summary of your post to LumaAI's Dream Machine model, which generates a five second video based on this input, which was a cute little addition. Try and imagine the story that resulted in this video: 

<video src="https://github.com/user-attachments/assets/42943e30-7969-449a-bfaf-b60d463b4e3e" controls="controls" width="100%"> </video>


Getting the whole thing up and running was surprisingly easy given that none of us had any experience with Streamlit (shout-out to Vindiw Wijesooriya who's [mistral-streamlit-chat repo](https://github.com/vindiw/mistral-streamlit-chat) pointed us in the right direction). 

The story writer worked pretty well off the bat, but getting the interviewer to work well was the biggest challenge. None of us had a lot of experience prompt engineering, and the interviewer kept repeating questions (like asking who you were with three times in a row), so messing with the interviewer prompt was one of the most time consuming steps. If you have ever been interviewed by a good interviewer you know that someone skilled at the craft of interviewing can steer a conversation in interesting directions, and that is the feeling we were going for here. Easier said than done. It definitely worked best when your initial response contains quite a bit of context, hence why our initial question to the user asks for at least a few sentences. Eventually we can up with something that gave a reasonable experience, but I feel there is a lot to be done there. Maybe coming up with a set list of questions, and only having the model do follow up questions? 


We made the story writer output the story it writes as markdown so they can easily be viewed and used as blog posts. As for the images we did something janky: we saved them as `image_n.jpg` where `n` is the order in which the images were uploaded, and told the model to refer to them in that way in markdown. This felt a little weird at first, but it seems to work well enough. The model will just dump a `![Picture description](picture_n.jpg)` in the markdown when it wanted to refer to the images, and it worked most of the time. 



## A few sample stories. 

In case you were wondering, no I didn't use JournalAIst to write this blog post. The language is clearly not flowery enough. To give you an idea of the quality of the writing, here are a few samples:

### Story 1: A Wild Rendezvous in Kruger National Park
I recounted my experience of being in the [Kruger National Park](https://en.wikipedia.org/wiki/Kruger_National_Park) with a couple of friends last year. For this case I basically just threw in a [handful of details](kruger-adventure/conversation_history.txt) and sent the model off to write. See the video at the end that we created while the LumaAI credits were still good. The video seems to try and combine a Lion, Elephant and Zebra, with the Elephant getting the Elephant's share of the mix. Also, note the abomination in the background.  

[Blog Post: Kruger Story 1](kruger-adventure/kruger-adventure.md)

Despite the minimal details, it did summarise what I told it quite truthfully, but with a healthy dash of breathless wonder, which doesn't sound like me. This might be as the prompt was write in the style of Bill Bryson, but even there I wouldn't say it sounds a lot like him either. The prompt also requests the model to write in a "humorous and engaging" manner, so I guess that might be the reason it sounds as it does. 

### Story 2: A Wild Ride through Kruger: A Tale of Elephants, Lions, and Fearless Honey Badgers

I tried giving it another go (caveat, this one was generated after the free credits expired, so we switched the model to `mistral-small-2409`), but this time added [more context](kruger-adventure2/conversation_history.txt) and chatted for a bit longer. 

[Blog Post: Kruger Story 2](kruger-adventure2/kruger-adventure2.md)

Again, the story is reasonable, although I wished it was maybe a bit longer? Or maybe it is mercifully short, depending on your perspective. 

### Story 3: A Weekend of Wonders: Our Hackathon Adventure

I also asked it to write up my experience of the hackathon with a few pictures I took during the event. I tried to continue the [interview for a while](hackathon-story/conversation_history.txt) to give as much info as possible. 

[Journal Entry: Hackathon Story](hackathon-story/hackathon-story.md)

The story is OK. Everything it stated was factual, and as the prompt this time was to write a journal entry and not a blog post, it's not as fantastical as before. It's a little short though, and I could likely have written the same if not more myself. 


## Conclusions

Our project was one of the few that actually fully worked by the end of the hackathon, and our first showing really impressed the judges (they told me afterwards). On the strength of that we made it to the final six teams (out of about 25 that decided to enter). The pitch went fine, and we again did a live demo in the four minutes allotted to us. We didn't end up winning any prizes, but we were just happy to be in the finals. Maybe we needed a salesperson on our team and not just a bunch of engineers 😉. 

I spoke to some of the judges afterwards, and one of the criticisms was that while initially impressive, they had worries about hallucinations: in our pitch we had highlighted that this is a way to hopefully tell an authentic and meaningful story, but with hallucinations, you might spend as much time fixing the story as you do talking to the model. This is extremely fair criticism and was also mirrored our worries. It did seem the longer you chatted to it the more factual the story ended up being (unsurprisingly), but at what point do you spend more time chatting then it would just have taken you to write it? 

On the last point: I would still argue that the job of a ghostwriter is to take your unstructured thoughts and edit them down into something coherent. 

My feeling is that the chat interface, or at least typing is not as free as I would have liked. One of the judges suggested a voice interface, which is something we considered as well (i.e. using something like [Whisper](https://github.com/openai/whisper) to turn speech into text), but would have been hard to get working in the little time we had. 


My feeling is that there is something interesting here, and that making the interaction with the interviewer more natural is probably where we should focus our efforts. 

The quality of the interviewer questions could also be better. Maybe we can prompt it by giving it some transcripts of interviews by Louis Theroux or some other good interviewer. 

## Code

After the hackathon the API keys provided expired, but Mistral recently opened up a [free tier](https://mistral.ai/news/september-24-release/), so we bumped the main model down to one of the smaller ones supported on the free tier. 

In any case, the code is [here](https://github.com/nyejon/journalaist) if you wish to play around with it. The code is what I would call "Hackathon Quality" so beware. You'll need a Mistral API key to run it, but it will work with the free tier, so won't cost you anything. 

It's also deployed on [Streamlit cloud](https://journalaist.streamlit.app/), which might work depending on whether or not we've hit the limits of Mistral's free tier. 


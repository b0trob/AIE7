# Vibe check check-list

1. Explain the concept of object-oriented programming in simple terms to a complete beginner.
    - Aspect Tested:
	    - It was simple enough for a beginner for understand? Yes
	    - It was well formatted? No
2. Read the following paragraph and provide a concise summary of the key points…
    - Aspect Tested:
	    - Instead of a paragraph I gave it a link to a github's documentation. It was able to reach out that source, i.e: the url. But then I asked to vistit a public repo of mine and it said that is not programmed to search on the web like that, which is odd.
	    - I also tested about its retention time, I asked about the request of point #1. It was able to hold the context just fine.
3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - Aspect Tested:
	    - Original? No, it returned a story about Wall.E
	    - Is the developer prompt being saved? I tested another developer prompt, since there is an option to select another from predefined ones, also you can write your own. But it did not work. The page needs to be refreshed to in order to take the new prompt, this is related to how it handles the chat sessions is will be improved in incoming versions.
4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - Aspect Tested:
	    - No verbosity, which is good, since I was asking for a simple answer. 
		- I also wanted to test the retention time at this point again, so I waited for about an hour and asked to give another summary of key points which was unable to do since it did not hold the context that long.
5. Rewrite the following paragraph in a professional, formal tone…
    - Aspect Tested:
	    - I tested to generate the informal paragraph first, the ask it to regenerate it in a formal tone. It worked as expected.

# Lessons learned:

- How to connect and interact with OpenAI API, while yet trivial it was an absolutely new accomplish for me.
- Without clear context, cursor will do exactly what is asked to do, for good or bad.
- By vibe checking, a lot of fun/unsuspected behavior can be detected before go full "PoC Show Time"

# Lessons yet to be learned:
- Properly understand and usage of `uv`
- How to make my chat to connect to whichever URL I give to it.
- The most efficient rules to develop a better one shot front-end based just on back-end code.

## Discussion question #1:
What are some limitations of vibe checking as an evaluation tool?

- It does not cover everything.
- It is slow.
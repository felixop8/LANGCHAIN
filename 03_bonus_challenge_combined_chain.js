import { config } from "dotenv";
config();
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
const outputParser = new StringOutputParser();

import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";


const model = new ChatOpenAI();

/* Bonus Challenge

    1. Create another chain called bonusChain, which takes the story sypnosis currently being
    generated by combinedChain and feeds it into the bonusPrompt supplied bellow.

    2. Add this chain to the combinedChain, so that when the chain is invoked, the end result is a
    string that contains the first few lines of dialogue for the story.
*/

const prompt1 = ChatPromptTemplate.fromTemplate(
    `What are three physical objects associated with the theme of {theme}? As an answer, provide just 
    a list of three comma-separated words.`
);

const prompt2 = ChatPromptTemplate.fromTemplate(
    `Provide a four-sentence sypnosis for a {genre} story that revolves around the following objects: {objects}.
    Give the story a title and a list of the objects at the beginning of the sypnosis. Reply in {language}.`
);

const bonusPrompt = ChatPromptTemplate.fromTemplate(
    `Provide the first four lines of dialogue for a story with the following sypnosis: {sypnosis}.`
);

const chain = prompt1.pipe(model).pipe(outputParser);

const bonusChain = RunnableSequence.from([ 
    {
        sypnosis: new RunnablePassthrough(),
    },
    bonusPrompt,
    model,
    outputParser
 ])

const combinedChain = RunnableSequence.from([
    {
        objects: chain,
        genre: (input) => input.genre,
        language: (input) => input.language
    },
    prompt2,
    model,
    outputParser,
    bonusChain
]);

const result = await combinedChain.invoke({
    theme: 'The life of a spanish inmigrant in the United States.',
    genre: "horror",
    language: "spanish"
});

console.log(result)
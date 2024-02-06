import { config } from "dotenv";
config();
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
const outputParser = new StringOutputParser();

import { RunnableSequence } from "@langchain/core/runnables";


const model = new ChatOpenAI();

/* Challenge

    1. Create a combinedChain that uses the two prompt bellow to generate a story,
    passing output generated via prompt1 into prompt2 and using the object already passed 
    into the invoke method as  the theme and genre variable values.
*/

const prompt1 = ChatPromptTemplate.fromTemplate(
    `What are three physical objects associated with the theme of {theme}? As an answer, provide just 
    a list of three comma-separated words.`
);

const prompt2 = ChatPromptTemplate.fromTemplate(
    `Provide a four-sentence sypnosis for a {genre} story that revolves around the following objects: {objects}.
    Give the story a title and a list of the objects at the beginning of the sypnosis.`
);

const chain = prompt1.pipe(model).pipe(outputParser);

const combinedChain = RunnableSequence.from([
    {
        objects: chain,
        genre: (input) => input.genre,
    },
    prompt2,
    model,
    outputParser
]);

const result = await combinedChain.invoke({
    theme: 'The life of a spanish conquistador.',
    genre: "horror"
});

console.log(result)
import { config } from "dotenv";
config();
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from "@langchain/core/runnables";
const outputParser = new StringOutputParser();
const model = new ChatOpenAI();

/*
    PromptTEmplate + LLM = ⛓ => invoke = input variable(s) => prompt template => prompt => mode => result
*/

const prompt1 =  ChatPromptTemplate.fromTemplate(
    `What's the city {person} is from? Only respond with the name of the city.`
);

const prompt2 =  ChatPromptTemplate.fromTemplate(
    `What country is the city {city} in? Reply in {language}.`
);

const chain = prompt1.pipe(model).pipe(outputParser);

const combinedChain = RunnableSequence.from([
    {
        city: chain,
        language: (input) => input.language
    },
    prompt2,
    model,
    outputParser
]);

const result = await combinedChain.invoke({
    person: 'Obama',
    language: "German"
});

console.log(result);




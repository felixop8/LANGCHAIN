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

const promptTemplate =  ChatPromptTemplate.fromTemplate(
    `Tell me a joke about {topic} in {language}?`
);

const chain = promptTemplate.pipe(model).pipe(outputParser);

await chain.invoke({
    topic: 'bears',
    language: 'english'
}).then(console.log);



// Challenge
const challengePromptTemplate = ChatPromptTemplate.fromTemplate(
    `Give me a {outputType} of {number} {type} facts about {subject}`
);

const challengeChain = challengePromptTemplate.pipe(model).pipe(outputParser);

await challengeChain.invoke({
    outputType: 'list',
    number: 5,
    type: 'interesting',
    subject: 'bears'
}).then(console.log);

import { config } from "dotenv";
config();
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI();


// 1. Create a vector store
const vectorStore = await MemoryVectorStore.fromTexts(
    [
        "mitochondria is the powerhouse of the cell",
        "mitochondria is made of lipids",
    ],
    [{ id: 1 }, { id: 2 }],
    new OpenAIEmbeddings(),
);
const retriever = vectorStore.asRetriever();
/////////////////////////////////////////////////////



// 2. Create condensed question chain
const condenseQuestionTemplate = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History: {chat_history}
  
Follow Up Input: {question}
Standalone question:
`;
const CONDENSE_QUESTION_PROMPT = PromptTemplate.fromTemplate(condenseQuestionTemplate);

const formatChatHistory = (chatHistory) => {
    const formattedDialogueTurns = chatHistory.map(
      (dialogueTurn) => `Human: ${dialogueTurn[0]}\nAssistant: ${dialogueTurn[1]}`
    );
    return formattedDialogueTurns.join("\n");
  };

const standaloneQuestionChain = RunnableSequence.from([
    {
        chat_history: (input) => formatChatHistory(input.chat_history),
        question: (input) => input.question
    },
    CONDENSE_QUESTION_PROMPT,
    model,
    new StringOutputParser()
]);
/////////////////////////////////////////////////////



// 3. Create answer chain
const answerTemplate = `Answer the question based only on the following context:
{context}

Question: {question}
`;
const ANSWER_PROMPT = PromptTemplate.fromTemplate(answerTemplate);
const serializeDocs = (docs, separator = "\n\n") => {
    const serializedDocs = docs.map((doc) => doc.pageContent);
    return serializedDocs.join(separator);
};
const answerChain = RunnableSequence.from([
    {
        context: retriever.pipe(serializeDocs),
        question: new RunnablePassthrough(),
    },
    ANSWER_PROMPT,
    model,
    new StringOutputParser()
]);
/////////////////////////////////////////////////////


// 4. Create conversational retrieval QA chain
const conversationalRetrievalQAChain = standaloneQuestionChain.pipe(answerChain);
/////////////////////////////////////////////////////


// 5. Invoke the conversational retrieval QA chain
await conversationalRetrievalQAChain.invoke({
    question: "What are they made out of?",
    chat_history: [
      [
        "What is the powerhouse of the cell?",
        "The powerhouse of the cell is the mitochondria.",
      ],
    ],
}).then(console.log);
/////////////////////////////////////////////////////
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
/* Retrieval-Augmented Generation 

    retrieval ⮕ prompt ⮕ model ⮕ parser
       ⬇   
    database 
  (vector store)
*/

const vectorStore = await MemoryVectorStore.fromTexts(
    [
        "mitochondria is the powerhouse of the cell",
        "lysosomes are the garbage disposal of the cell",
        "the nucleus is the control center of the cell",
    ],
    [{ id: 1 }, { id: 2 }, { id: 3 }],
    new OpenAIEmbeddings(),
)

const retriever = vectorStore.asRetriever()

const serializeDocs = (docs) => docs.map((doc) => doc.pageContent).join("\n");

const prompt =
  PromptTemplate.fromTemplate(`Answer the question based only on the following context:
{context}

Question: {question}`);

const chain = RunnableSequence.from([
    {
        context: retriever.pipe(serializeDocs),
        question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser()
])

await chain.invoke("What is the powerhouse of the cell?").then(console.log)


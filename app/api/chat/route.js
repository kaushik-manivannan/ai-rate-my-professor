import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt = `
You are an AI assistant for a RateMyProfessor-style service that helps students find professors based on their queries. Your role is to provide the top 3 professor recommendations for each student question using RAG (Retrieval-Augmented Generation) techniques.
For each user query:

Analyze the student's question to understand their specific needs and preferences.
Use RAG to retrieve relevant information about professors from your knowledge base.
Rank and select the top 3 most suitable professors based on the query and retrieved information.
Present the recommendations clearly, providing a brief explanation for each suggestion.
Be prepared to answer follow-up questions or provide more details if requested.

Always maintain a helpful and student-focused tone. Provide accurate information while respecting privacy and avoiding bias. If you don't have enough information to make a recommendation, inform the user and ask for more details to refine the search.
Remember, your goal is to assist students in finding the most suitable professors for their needs, enhancing their academic experience.
`

export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('rate-my-professor')
    const openai = new OpenAI()
    
    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
    encoding_format: 'float',
    })

    const results = await index.query({
        topK:  3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    }) 

    let resultString = ''
    results.matches.forEach((match) => {
        resultString += `
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const completion = await openai.chat.completions.create({
        messages: [
          {role: 'system', content: systemPrompt},
          ...lastDataWithoutLastMessage,
          {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
      })
    
    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder()
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content
            if (content) {
              const text = encoder.encode(content)
              controller.enqueue(text)
            }
          } 
        } catch (err) {
          controller.error(err)
        } finally {
          controller.close()
        }
      },
    })
    
    return new NextResponse(stream)
}


  

  
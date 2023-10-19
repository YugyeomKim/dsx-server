// @ts-check

const express = require('express')
const fs = require('fs')
const path = require('path')
const cors = require('cors')
const { spawn } = require('child_process')

const { getUsersCollection, getStatCollection } = require('./mongo')

const app = express()
const PORT = 3000

const BUFFER_PATH = path.join(__dirname, '../buffer')

try {
  console.log(BUFFER_PATH)
  fs.readdirSync(BUFFER_PATH)
} catch (err) {
  console.error(`${BUFFER_PATH} does not exist! Creating...`)
  fs.mkdirSync(BUFFER_PATH)
}

app.use(cors())
app.use(express.json({ limit: '10gb' }))

app.use('/predict', async (req, res) => {
  const startTime = Date.now()

  /** @type {string[]} */
  const data = req.body

  if (!data) {
    res.status(400).send('No image data found')
    return
  }

  console.log(`Received image in ${Date.now() - startTime}ms`)

  const imageNamePrefix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`
  /** @type {string[]} imageList of saved images with no error when writing */
  const imagePathList = await Promise.all(
    data.map((base64, idx) => {
      const imageName = `${imageNamePrefix}-${idx}.jpg`
      const imagePath = path.join(BUFFER_PATH, imageName)
      const img = Buffer.from(base64, 'base64')
      return new Promise((resolve, reject) => {
        fs.writeFile(imagePath, img, (err) => {
          if (err) {
            console.log(err)
          } else {
            resolve(imagePath)
          }
        })
      })
    })
  )

  console.log(`Write image in ${Date.now() - startTime}ms`)

  const python = spawn('python3', [
    'model/run-model.py',
    JSON.stringify(imagePathList),
  ])

  python.stdout.on(
    'data',
    /** @param {number} data */ (data) => {
      console.log(`Got data from Python script in ${Date.now() - startTime}ms`)

      // Get data from Python script
      const prediction = JSON.parse(data.toString())

      // Delete image
      imagePathList.forEach((imagePath) => {
        fs.unlink(imagePath, (err) => {
          if (err) console.log(`${imagePath} not found!`)
        })
      })

      // Send prediction back to client
      res.send(prediction)
    }
  )
})

app.use('/survey', async (req, res) => {
  const { type, email, ...userData } = req.body

  /** @type {{ stat: boolean | null, user: boolean | null }} */
  const result = {
    stat: null,
    user: null,
  }

  if (type && type === 'running') {
    const stat = await getStatCollection()
    if (!stat) {
      res.statusCode = 404
      res.send('MongoDB connection failed.')

      return
    }

    /** @type {import('mongodb').UpdateResult<Document>} */
    const updateResult = await stat.updateOne(
      {},
      {
        $inc: {
          'total-run': 1,
        },
      }
    )

    result.stat = updateResult.acknowledged
  }

  const users = await getUsersCollection()
  if (!users) {
    res.statusCode = 404
    res.send('MongoDB connection failed.')

    return
  }

  /** @type {Object.<string, Object<string, string>>} */
  const update = { $set: {} }
  Object.entries(userData).forEach((entry) => {
    update.$set[entry[0]] = entry[1]
  })

  const updateResult = await users.updateOne(
    {
      email: email,
    },
    update,
    {
      upsert: true,
    }
  )

  result.user = updateResult.acknowledged
  res.send(result)
})

app.listen(PORT, () => {
  console.log(`Listening at http://localhost:${PORT}`)
})

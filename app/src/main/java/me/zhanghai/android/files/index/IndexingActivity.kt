package me.zhanghai.android.files.index

import android.os.Bundle
import me.zhanghai.android.files.app.AppActivity
import me.zhanghai.android.files.databinding.IndexingActivityBinding

class IndexingActivity : AppActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val binding = IndexingActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        supportActionBar!!.setDisplayHomeAsUpEnabled(true)
    }
}
